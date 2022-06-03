import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import ViTConfig, ViTForImageClassification
import argparse
from tqdm import tqdm
import numpy as np
import random
import os
import time
from data import build_dataset
from config import get_config

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../site-package')
from utils import print_peak_memory
from megatron.initialize import initialize_megatron
from megatron import get_args, _print_args
from torch.nn.parallel import DistributedDataParallel as DDP
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.pipeline.sync import Pipe
from utils import gen_groups, show_groups, modules_to_devices, wrap_modules_data_parallel, wrap_modules_relocation

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def overwrite_megatron_args(config, args):
    args.hidden_size = config.hidden_size
    args.num_layers = config.num_hidden_layers
    args.num_attention_heads = config.num_attention_heads
    args.ffn_hidden_size = config.intermediate_size
    args.max_position_embeddings = (config.image_size // config.patch_size) ** 2
    args.attention_dropout = config.attention_probs_dropout_prob
    args.hidden_dropout = config.hidden_dropout_prob

def get_chunks(args):
    if args.chunks == -1:
        args.chunks = 1
        if args.global_tp_deg > 0 and args.pp_deg > 1:
            world_size = torch.distributed.get_world_size()
            global_dp_deg = world_size // args.global_tp_deg
            bsz_per_gpu = args.global_train_batch_size/global_dp_deg 
            if bsz_per_gpu == 8:
                optimal_micro_bsz = 1
            else:
                optimal_micro_bsz = 2
            args.chunks = np.ceil(optimal_micro_bsz)
    if torch.distributed.get_rank() == 0:
        print('Chunks:', args.chunks)
    return args.chunks

def construct_hybrid_parallel_model(vit_model, config, tp_sizes_enc, dp_types_enc, pp_ranks_enc, pp_deg, tp_consecutive_flags):
    assert config.num_hidden_layers == len(tp_sizes_enc)
    assert config.num_hidden_layers == len(dp_types_enc) 
    assert config.num_hidden_layers == len(pp_ranks_enc)
    world_size = torch.distributed.get_world_size()
    for tp_size in tp_sizes_enc:
        assert tp_size <= world_size and world_size % tp_size == 0 and tp_size >= 1, 'Wrong tp_size!'
    for dp_type in dp_types_enc:
        assert dp_type == 0 or dp_type == 1 or dp_type is None, 'Wrong dp_type!'
    for pp_rank in pp_ranks_enc:
        assert pp_rank >= 0 and pp_rank <= pp_deg - 1, 'Wrong pp_rank!'
    
    # [Step 0] Construct sizes & groups
    # Construct tp_sizes / dp_types / pp_stages for whole model
    tp_sizes_whole_model  = [1] + tp_sizes_enc + [1, 1]
    dp_types_whole_model  = [0] + dp_types_enc + [0, 0]
    pp_ranks_whole_model = [0] + pp_ranks_enc + [pp_deg-1, 0]
    tp_consecutive_whole_model = [1] + tp_consecutive_flags + [1, 1]
    # Construct tp_groups / dp_groups / allgather_groups / slice_funcs /label_relocation_func
    tp_groups_whole_model, dp_groups_whole_model, allgather_groups_whole_model, slice_funcs_whole_model = gen_groups(tp_sizes_whole_model, tp_consecutive_whole_model, show_rank = 0)
    tp_groups_enc = tp_groups_whole_model[1:-2]

    # [Step 1] Construct Tensor Parallel Block based on tp_groups
    from ViTForImageClassification_tensor_parallel import VitEncoder_tp
    encoder_tp = VitEncoder_tp(config, tp_groups=tp_groups_enc)
    setattr(vit_model.vit, 'encoder', encoder_tp)

    # [Step 2] Construct Sequantial modules
    from ViTForImageClassification_pipeline import VitClassification_, VitEmbedding_, VitEncoder_, VitLayerNorm_
    model = nn.Sequential()
    model.add_module('embedding', VitEmbedding_(vit_model))
    for i in range(config.num_hidden_layers):
        model.add_module('encoder_%d'%(i), VitEncoder_(vit_model, i, i + 1))
    model.add_module('layernorm', VitLayerNorm_(vit_model))
    model.add_module('cls', VitClassification_(vit_model))

    # [Step 3] Wrap Relocation modules if necessary
    model = wrap_modules_relocation(model, allgather_groups_whole_model, slice_funcs_whole_model)

    # [Step 4] Place Sequantial modules to GPU devices based on pp_stages
    rank = torch.distributed.get_rank()
    devices = [i * world_size + rank for i in range(pp_deg)]
    pp_devices_whole_model = [devices[i] for i in pp_ranks_whole_model]
    modules_to_devices(model, pp_devices_whole_model)

    # [Step 5] Wrap Data Parallel modules based on dp_types & dp_groups
    module_types = ['embed'] + ['vit_enc']*config.num_hidden_layers + ['layernorm', 'cls']
    model = wrap_modules_data_parallel(model, dp_types_whole_model, dp_groups_whole_model, module_types=module_types, pp_devices=pp_devices_whole_model)

    # [Step 6] Construct Pipeline Parallel model
    chunks = get_chunks(args)
    model_hp = Pipe(model, chunks=chunks, checkpoint='never')
    return model_hp

def apply_layerwise_hybrid_strategy():
    ###### ViT Huge 32 layers 1280 hidden
    ### A sample strategy
    pp_deg = 1
    tp_sizes_enc = [1]*32
    tp_consecutive_flags = [1]*32
    dp_types_enc = [0]*20+[1]*12

    if torch.distributed.get_rank() == 0:
        print('tp_sizes:', tp_sizes_enc)
        print('tp_consec:',tp_consecutive_flags)
        print('dp_type:',dp_types_enc)
        print('pp_deg:',pp_deg)
    return tp_sizes_enc, tp_consecutive_flags, dp_types_enc, pp_deg

def train(args, conf):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()
    pp_deg = args.pp_deg
    assert pp_deg * world_size <= 8, 'pp_deg * world_size should <= 8!'
    assert args.global_train_batch_size % world_size == 0, 'global_train_batch_size should be multiple of world_size!'
    train_batch_size_input = args.global_train_batch_size // world_size
    if args.global_tp_deg > 0:
        assert args.global_tp_deg <= world_size
        if rank == 0:
            print('Strategy:', args.pp_deg, args.global_tp_deg, world_size//args.global_tp_deg)

    import torch.distributed.rpc as rpc
    rpc.init_rpc(
            name="worker%d" % rank,
            rank = rank,
            world_size=world_size,
    )

    print("Creating Dataloader...")
    dataset, num_classes = build_dataset(is_train=True, config=conf)
    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input,
                            shuffle=False)

    print("Creating Model...")
    config = ViTConfig(hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers, 
                        num_attention_heads=args.num_attention_heads, 
                        intermediate_size=args.hidden_size*4, 
                        attention_probs_dropout_prob=args.dropout_prob,
                        hidden_dropout_prob=args.dropout_prob,
                        image_size=args.image_size,
                        patch_size=args.patch_size,
                        num_channels=args.num_channels)
    config.num_labels = num_classes
    overwrite_megatron_args(config, args)
    vit_model = ViTForImageClassification(config)

    if args.apply_strategy:
        tp_sizes_enc, tp_consecutive_flags, dp_types_enc, pp_deg = apply_layerwise_hybrid_strategy()
    else:
        assert(args.global_tp_deg > 0 and args.global_tp_consec in [0, 1])
        tp_sizes_enc = [args.global_tp_deg] * args.num_hidden_layers
        tp_consecutive_flags = [args.global_tp_consec] * args.num_hidden_layers
        dp_types_enc = args.num_hidden_layers * [args.fsdp]

    avg_num_layers = config.num_hidden_layers // pp_deg
    pp_ranks_enc = []
    for i in range(pp_deg):
        pp_ranks_enc += [i] * avg_num_layers

    model = construct_hybrid_parallel_model(vit_model, config, tp_sizes_enc, dp_types_enc, pp_ranks_enc, pp_deg, tp_consecutive_flags)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    profile_rank = 0
    if args.profile and rank == profile_rank:
        print_peak_memory("After creating model", rank, args.profile_type)

    start_iter, end_iter = 10, 20
    print("Start training...")
    for ep in range(args.epochs):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader)
        for iter, batch in enumerate(trainloader):
            start_time = time.time()
            if args.profile:
                if iter == start_iter:
                    total_start_time = start_time
                elif iter == end_iter:
                    total_end_time = start_time
                    avg_time = (total_end_time-total_start_time)/(end_iter-start_iter)
                    print("Average iteration time is: %.4f s"%avg_time)
                    return
            input, label= [tensor.to(device) for tensor in batch]

            if args.profile and rank == profile_rank and iter <= 2:
                torch.cuda.reset_peak_memory_stats(rank)
                print_peak_memory("\nBefore Forward", rank, args.profile_type)

            # model forward
            outputs = model(input).local_value()
            lossft = nn.CrossEntropyLoss()
            loss = lossft(outputs.view(-1, config.num_labels), label.view(-1))

            if args.profile and rank == profile_rank and iter <= 2:
                print_peak_memory("After Forward", rank, args.profile_type)

            loss.backward()

            if args.profile and rank == profile_rank and iter <= 2:
                print_peak_memory("After Backward", rank, args.profile_type)
            
            optimizer.step()

            if args.profile and rank == profile_rank and iter <= 2:
                print_peak_memory("After optimizer_step", rank, args.profile_type)
            
            optimizer.zero_grad()

            end_time = time.time()
            if args.check_loss or args.profile:
                print('[Epoch %d] (Iteration %d): Loss = %.6f, Time = %.3f'% (ep,iter,loss.item(), end_time-start_time))

def add_arguments(parser):
    group = parser.add_argument_group(title='our arguments')

    group.add_argument(
        "--global_train_batch_size", type=int, default=32, help="Training batch size for single GPU"
    )
    group.add_argument(
        "-s", "--seq_length", type=int, default=196, help="Maximum sequence len"
    )
    group.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        '--data-folder', default = 'ImageNet', type=str, help='path to dataset'
    )
    group.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    group.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    group.add_argument(
        "--image_size", type=int, default=224, help="Input image size."
    )
    group.add_argument(
        "--patch_size", type=int, default=16, help="Patch size of ViT."
    )
    group.add_argument(
        "--num_channels", type=int, default=3, help="Number of channels."
    )
    group.add_argument(
        "--dropout_prob", type=float, default=0.0, help="Dropout rate."
    )
    group.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    group.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    group.add_argument(
        '--zip', type=bool, default=True, help='use zipped dataset instead of folder dataset'
    )
    group.add_argument(
        '--cache-mode', type=str, default='no', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece'
    )
    group.add_argument(
        "--check_loss", type=int, default=0, help="Whether to check model correctness."
    )
    group.add_argument(
        "--profile", type=int, default=0, help="Whether to profile model GPU memory."
    )
    group.add_argument(
        "--profile_type", type=str, default='allocated', help="Profile allocated memory or reserved memory.",
        choices = ['allocated', 'reserved'],
    )
    parser.add_argument(
        "--load_params", type=int, default=0, help="Whether to load saved init params."
    )
    parser.add_argument(
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8],
    )
    parser.add_argument(
        "--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8],
    )
    parser.add_argument(
        "--chunks", type=int, default=-1, help="Pipeline chunk num.",
    )
    parser.add_argument(
        "--global_tp_consec", type=int, default=-1, help="Global tensor parallel group consecutive flag."
    )
    parser.add_argument(
        "--fsdp", type=int, default=0, help="Apply FSDP", choices=[0, 1],
    )
    parser.add_argument(
        "--apply_strategy", type=int, default=0, help="Apply searched strategy.", choices=[0, 1],
    )

    return parser

if __name__ == '__main__':
    initialize_megatron(extra_args_provider=add_arguments)
    args = get_args()
    config = get_config(args)
    set_seed()

    train(args, config)