from email import generator
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import SwinConfig, SwinForImageClassification
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
    args.num_layers = sum(config.depths)
    args.num_attention_heads = config.num_heads
    args.max_position_embeddings = config.embed_dim
    args.attention_dropout = config.attention_probs_dropout_prob
    args.hidden_dropout = config.hidden_dropout_prob

def get_chunks(args):
    if args.chunks == -1:
        args.chunks = 1
        if args.global_tp_deg > 0 and args.pp_deg > 1:
            world_size = torch.distributed.get_world_size()
            global_dp_deg = world_size // args.global_tp_deg
            bsz_per_gpu = args.global_train_batch_size/global_dp_deg 
            if bsz_per_gpu <= 8:
                optimal_micro_bsz = 1
            elif bsz_per_gpu > 8 and bsz_per_gpu < 32:
                optimal_micro_bsz = 2
            elif bsz_per_gpu >= 32 and bsz_per_gpu <= 96:
                optimal_micro_bsz = 3
            else:
                optimal_micro_bsz = 4
            args.chunks = np.ceil(optimal_micro_bsz)
    if torch.distributed.get_rank() == 0:
        print('Chunks:', args.chunks)
    return args.chunks

def construct_hybrid_parallel_model(swin_model, config, tp_sizes_enc, dp_types_enc, pp_stages_enc, pp_deg, tp_consecutive_flags):
    num_hidden_layers = sum(config.depths)
    
    assert num_hidden_layers == len(tp_sizes_enc)
    assert num_hidden_layers == len(dp_types_enc) 
    assert num_hidden_layers == len(pp_stages_enc)
    world_size = torch.distributed.get_world_size()
    for tp_size in tp_sizes_enc:
        assert tp_size <= world_size and world_size % tp_size == 0 and tp_size >= 1, 'Wrong tp_size!'

    # [Step 0] Construct sizes & groups
    # Construct tp_sizes / dp_types / pp_stages for whole model
    tp_sizes_whole_model  = [1] + tp_sizes_enc + [1, 1]
    dp_types_whole_model  = [0] + dp_types_enc + [0, 0]
    pp_stages_whole_model = [0] + pp_stages_enc + [pp_deg-1, pp_deg-1]
    tp_consecutive_whole_model = [1] + tp_consecutive_flags + [1, 1]
    # Construct tp_groups / dp_groups / allgather_groups / slice_funcs /label_relocation_func
    tp_groups_whole_model, dp_groups_whole_model, allgather_groups_whole_model, slice_funcs_whole_model = gen_groups(tp_sizes_whole_model, tp_consecutive_whole_model, show_rank=0)
    tp_groups_enc = tp_groups_whole_model[1:-2]

    # [Step 1] Construct Tensor Parallel Block based on tp_groups
    from SwinForImageClassification_tensor_parallel import build_swinblock_list
    gen = tp_groups_enc.__iter__()
    for i, swinlayer in enumerate(swin_model.swin.encoder.layers):
        new_layers = build_swinblock_list(swinlayer.config, swinlayer.dim, swinlayer.blocks[0].input_resolution, 
            swinlayer.config.depths[i], swinlayer.config.num_heads[i], gen)
        setattr(swinlayer, 'blocks', new_layers)
    
    # [Step 2] Construct Sequantial modules
    from SwinForImageClassification_pipeline import SwinCls_, SwinEmbeddings_, SwinLayernorm_, SwinBlock_
    model = nn.Sequential()
    model.add_module('embeddings', SwinEmbeddings_(swin_model))

    for i, d in enumerate(args.depths):
        for j in range(d):
            model.add_module('encoder_%d_%d'%(i, j), SwinBlock_(swin_model, i, j, j==d-1))
        
    model.add_module('layernorm', SwinLayernorm_(swin_model))
    model.add_module('cls', SwinCls_(swin_model))

    # [Step 3] Wrap Relocation modules if necessary
    model = wrap_modules_relocation(model, allgather_groups_whole_model, slice_funcs_whole_model)

    # [Step 4] Place Sequantial modules to GPU devices based on pp_stages
    rank = torch.distributed.get_rank()
    devices = [i * world_size + rank for i in range(pp_deg)]
    pp_devices_whole_model = [devices[i] for i in pp_stages_whole_model]
    modules_to_devices(model, pp_devices_whole_model)

    module_types = ['embed'] + ['swin_enc']*num_hidden_layers + ['pooler', 'cls']
    # [Step 5] Wrap Data Parallel modules based on dp_types & dp_groups
    model = wrap_modules_data_parallel(model, dp_types_whole_model, dp_groups_whole_model, module_types, pp_devices=pp_devices_whole_model)

    # [Step 6] Construct Pipeline Parallel model
    chunks = get_chunks(args)
    model_hp = Pipe(model, chunks=chunks, checkpoint='never')
    return model_hp

def apply_layerwise_hybrid_strategy():
    ###### Swin Huge 32 layers 1280 hidden
    ### A sample strategy
    pp_deg = 1
    tp_sizes_enc = [1]*30+[2]*2
    tp_consecutive_flags = [1]*32
    dp_types_enc = [1]*2+[0]*2+[1]*28

    if torch.distributed.get_rank() == 0:
        print('tp_sizes:', tp_sizes_enc)
        print('tp_consec:',tp_consecutive_flags)
        print('dp_type:',dp_types_enc)
        print('pp_deg:',pp_deg)
    return tp_sizes_enc, tp_consecutive_flags, dp_types_enc, pp_deg

# Swin Huge 32 pp divide
pp_stage_dict_for_bsz = {8: {1: [32], 2: [12, 20], 4: [5, 10, 10, 7], 8: [1, 4, 5, 5, 5, 5, 5, 2]}, 
                        16: {1: [32], 2: [12, 20], 4: [4, 10, 10, 8], 8: [1, 4, 5, 5, 5, 5, 5, 2]}, 
                        24: {1: [32], 2: [11, 21], 4: [4, 10, 10, 8], 8: [1, 3, 5, 5, 5, 5, 5, 3]}, 
                        32: {1: [32], 2: [11, 21], 4: [3, 10, 10, 9], 8: [1, 2, 5, 5, 5, 5, 5, 4]}, 
                        40: {1: [32], 2: [11, 21], 4: [3, 10, 10, 9], 8: [1, 2, 5, 5, 5, 5, 5, 4]}, 
                        48: {1: [32], 2: [11, 21], 4: [3, 10, 10, 9], 8: [1, 2, 5, 5, 5, 5, 5, 4]}, 
                        56: {1: [32], 2: [11, 21], 4: [3, 10, 10, 9], 8: [1, 2, 5, 5, 5, 5, 5, 4]}, 
                        64: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 5, 5, 5, 5, 5, 4]}, 
                        72: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        80: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        88: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        96: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        104: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        112: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        120: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        128: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        136: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        144: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        152: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        160: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        168: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        176: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        184: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        192: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        200: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}}

# # Swin Huge 48 pp divide
# pp_stage_dict_for_bsz = {8: {1: [48], 2: [20, 28], 4: [7, 15, 15, 11], 8: [3, 7, 7, 7, 7, 7, 7, 3]}, 
#                         16: {1: [48], 2: [19, 29], 4: [8, 14, 14, 12], 8: [2, 7, 7, 7, 7, 7, 7, 4]}, 
#                         24: {1: [48], 2: [19, 29], 4: [8, 14, 14, 12], 8: [2, 6, 7, 7, 7, 7, 7, 5]}, 
#                         32: {1: [48], 2: [19, 29], 4: [7, 14, 14, 13], 8: [2, 5, 7, 7, 7, 7, 7, 6]}, 
#                         40: {1: [48], 2: [19, 29], 4: [7, 14, 14, 13], 8: [2, 5, 7, 7, 7, 7, 7, 6]}, 
#                         48: {1: [48], 2: [19, 29], 4: [7, 14, 14, 13], 8: [2, 5, 7, 7, 7, 7, 7, 6]}, 
#                         56: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 5, 7, 7, 7, 7, 7, 6]}, 
#                         64: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 5, 7, 7, 7, 7, 7, 6]}, 
#                         72: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
#                         80: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
#                         88: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
#                         96: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
#                         104: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
#                         112: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]},
#                         120: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}}

def get_pp_ranks_enc(global_bsz, pp_deg):
    pp_ranks_enc = []
    pp_divide = pp_stage_dict_for_bsz[global_bsz][pp_deg]
    for i in range(pp_deg):
        pp_ranks_enc += [i]*pp_divide[i]
    if torch.distributed.get_rank() == 0:
        print(pp_divide)
        print(pp_ranks_enc)
    return pp_ranks_enc


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
    config = SwinConfig(drop_path_rate=args.drop_path_rate,
                       embed_dim=args.embed_dim,
                       depths=args.depths,
                       num_heads=args.num_heads,
                       window_size=args.window_size)
    config.num_labels = num_classes
    overwrite_megatron_args(config, args)

    swin_model = SwinForImageClassification(config)

    if args.apply_strategy:
        tp_sizes_enc, tp_consecutive_flags, dp_types_enc, pp_deg = apply_layerwise_hybrid_strategy()
    else:
        assert(args.global_tp_deg > 0 and args.global_tp_consec in [0, 1])
        tp_sizes_enc = [args.global_tp_deg] * args.num_layers
        tp_consecutive_flags = [args.global_tp_consec] * args.num_layers
        dp_types_enc = args.num_layers * [args.fsdp]

    # # divide pipeline stage averagely according to layer number, may cause imbalanced workload
    # dp_types_enc = [args.fsdp] *  args.num_layers
    # avg_num_layers = args.num_layers // args.pp_deg
    # remains =  args.num_layers % args.pp_deg
    # pp_ranks_enc = []
    # for i in range(pp_deg):
    #     pp_ranks_enc += [i] * (avg_num_layers + (1 if i < remains else 0))

    # divide pipeline stage averagely according to memory workload
    pp_ranks_enc = get_pp_ranks_enc(args.global_train_batch_size, pp_deg)

    model = construct_hybrid_parallel_model(swin_model, config, tp_sizes_enc, dp_types_enc, pp_ranks_enc, args.pp_deg, tp_consecutive_flags)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)
    
    start_iter, end_iter = 10, 20
    profile_rank = 0
    if args.profile and rank == profile_rank:
        print_peak_memory("After creating model", rank, args.profile_type)

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
            input = batch[0].to(device)

            if args.profile and rank == profile_rank and iter <= 2:
                torch.cuda.reset_peak_memory_stats(rank)
                print_peak_memory("\nBefore Forward", rank, args.profile_type)

            # model forward
            logits = model(input).local_value()
            label = batch[1].to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_classes), label.view(-1))

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
            if (args.check_loss or args.profile) and rank == profile_rank:
                print('[Epoch %d] (Iteration %d): Loss = %.6f, Time = %.3f'% (ep,iter,loss.item(), end_time-start_time))

def add_arguments(parser):
    group = parser.add_argument_group(title='our arguments')

    group.add_argument(
        "--drop_path_rate", type=float, default=0.2, help="Drop path rate."
    )
    group.add_argument(
        "--global_train_batch_size", type=int, default=32, help="Training batch size for single GPU."
    )
    group.add_argument(
        "--embed_dim", type=int, default=12, help="Embed dim.",
    )
    group.add_argument(
        "--depths", nargs='+', type=int, default=[1], help="Depths."
    )
    group.add_argument(
        "--num_heads", nargs='+', type=int, default=[2], help="Num heads."
    )
    group.add_argument(
        "--window_size", type=int, default=7, help="Window size."
    )
    group.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs.")
    group.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam."
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
    group.add_argument(
        "--data-folder", default = 'ImageNet', type=str, help="Path to dataset."
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
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    
    parser.add_argument(
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8],
    )
    parser.add_argument(
        "--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8],
    )
    parser.add_argument(
        "--global_tp_consec", type=int, default=-1, help="Global tensor parallel group consecutive flag."
    )
    parser.add_argument(
        "--chunks", type=int, default=-1, help="Pipeline chunk num.",
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