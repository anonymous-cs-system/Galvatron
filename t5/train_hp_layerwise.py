import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import T5ForConditionalGeneration, T5Config
from transformers.optimization import Adafactor
from dataloader import DataLoaderForT5
import argparse
from tqdm import tqdm
import numpy as np
import random
import time
import os
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
    args.hidden_size = config.d_model
    args.num_layers = config.num_layers + config.num_decoder_layers
    args.num_attention_heads = config.num_heads
    args.ffn_hidden_size = config.d_ff
    args.max_position_embeddings = config.n_positions
    args.attention_dropout = config.dropout_rate
    args.hidden_dropout = config.dropout_rate

def get_chunks(args):
    if args.chunks == -1:
        args.chunks = 1
        if args.global_tp_deg > 0 and args.pp_deg > 1:
            world_size = torch.distributed.get_world_size()
            global_dp_deg = world_size // args.global_tp_deg
            bsz_per_gpu = args.global_train_batch_size/global_dp_deg 
            if bsz_per_gpu == 2:
                optimal_micro_bsz = 1
            elif args.pp_deg == 8:
                optimal_micro_bsz = 3
            else:
                optimal_micro_bsz = 2
            args.chunks = np.ceil(optimal_micro_bsz)
    if torch.distributed.get_rank() == 0:
        print('Chunks:', args.chunks)
    return args.chunks

def construct_hybrid_parallel_model(t5_model, config, tp_sizes_enc, dp_types_enc, pp_ranks_enc, pp_deg, tp_consecutive_flags):
    assert config.num_layers + config.num_decoder_layers == len(tp_sizes_enc)
    assert config.num_layers + config.num_decoder_layers == len(dp_types_enc) 
    assert config.num_layers + config.num_decoder_layers == len(pp_ranks_enc)
    world_size = torch.distributed.get_world_size()
    for tp_size in tp_sizes_enc:
        assert tp_size <= world_size and world_size % tp_size == 0 and tp_size >= 1, 'Wrong tp_size!'
    for dp_type in dp_types_enc:
        assert dp_type == 0 or dp_type == 1 or dp_type is None, 'Wrong dp_type!'
    for pp_rank in pp_ranks_enc:
        assert pp_rank >= 0 and pp_rank <= pp_deg - 1, 'Wrong pp_rank!'

    # [Step 0] Construct sizes & groups
    # Construct tp_sizes / dp_types / pp_stages for whole model
    tp_sizes_whole_model  = [1] + tp_sizes_enc[:config.num_layers] + [1] + tp_sizes_enc[config.num_layers:] + [1]
    dp_types_whole_model  = [0] + dp_types_enc[:config.num_layers] + [0] + dp_types_enc[config.num_layers:] + [0]
    pp_stages_whole_model = [0] + pp_ranks_enc[:config.num_layers] + [0] + pp_ranks_enc[config.num_layers:] + [0]
    tp_consecutive_whole_model = [1] + tp_consecutive_flags[:config.num_layers] + [1] + tp_consecutive_flags[config.num_layers:] + [1]
    # Construct tp_groups / dp_groups / allgather_groups / slice_funcs /label_relocation_func
    tp_groups_whole_model, dp_groups_whole_model, allgather_groups_whole_model, slice_funcs_whole_model = gen_groups(tp_sizes_whole_model, tp_consecutive_whole_model, show_rank = 0)
    tp_groups_enc = tp_groups_whole_model[1:-2]

    # [Step 1] Construct Tensor Parallel Block based on tp_groups
    from T5ForConditionalGeneration_tensor_parallel import T5LayerFF_tp, T5Attention_tp, T5Block_tp
    from T5ForConditionalGeneration_tensor_parallel import get_extended_attention_mask_encoder, get_extended_attention_mask_decoder, invert_attention_mask

    self_config = t5_model.encoder.config
    for i in range(config.num_layers):
        layer = t5_model.encoder.block[i].layer
        setattr(layer[0], 'SelfAttention', T5Attention_tp(self_config, tp_group=tp_groups_enc[i]))
        layer[-1] = T5LayerFF_tp(self_config, tp_group=tp_groups_enc[i])
        setattr(t5_model.encoder.block[i], 'layer', layer)
        t5_model.encoder.block[i] = T5Block_tp(t5_model.encoder.block[i])
    setattr(t5_model.encoder, 'get_extended_attention_mask', get_extended_attention_mask_encoder)

    cross_config = t5_model.decoder.config
    for i in range(config.num_decoder_layers):
        layer = t5_model.decoder.block[i].layer
        setattr(layer[0], 'SelfAttention', T5Attention_tp(self_config, tp_group=tp_groups_enc[i+config.num_layers]))
        setattr(layer[1], 'EncDecAttention', T5Attention_tp(cross_config, tp_group=tp_groups_enc[i+config.num_layers]))
        layer[-1] = T5LayerFF_tp(cross_config, tp_group=tp_groups_enc[i+config.num_layers])
        setattr(t5_model.decoder.block[i], 'layer', layer)
        t5_model.decoder.block[i] = T5Block_tp(t5_model.decoder.block[i])
    setattr(t5_model.decoder, 'get_extended_attention_mask', get_extended_attention_mask_decoder)
    setattr(t5_model.decoder, 'invert_attention_mask', invert_attention_mask)

    # [Step 2] Construct Sequantial modules
    from T5ForConditionalGeneration_pipeline import T5Embeddings_, T5Decoder_, T5Cls_, T5Encoder_, T5DecoderEmbedding_
    model = nn.Sequential()
    model.add_module('embeddings_1', T5Embeddings_(t5_model))
    for i in range(config.num_layers):
        model.add_module('encoder_%d'%(i), 
            T5Encoder_(t5_model, i, has_final_layernorm= i + 1 >= config.num_layers))

    model.add_module('embeddings_2', T5DecoderEmbedding_(t5_model))
    for i in range(config.num_decoder_layers):
        model.add_module('decoder_%d'%(i), 
            T5Decoder_(t5_model, i, has_final_layernorm= i + 1 >= config.num_decoder_layers))
    model.add_module('cls', T5Cls_(t5_model))

    # [Step 3] Wrap Relocation modules if necessary
    model = wrap_modules_relocation(model, allgather_groups_whole_model, slice_funcs_whole_model)

    # [Step 4] Place Sequantial modules to GPU devices based on pp_stages
    rank = torch.distributed.get_rank()
    devices = [i * world_size + rank for i in range(pp_deg)]
    pp_devices_whole_model = [devices[i] for i in pp_stages_whole_model]
    modules_to_devices(model, pp_devices_whole_model)

    # [Step 5] Wrap Data Parallel modules based on dp_types & dp_groups
    module_types = ['embed_1'] + ['t5_enc']*config.num_layers + ['embed_2'] + ['t5_dec']*config.num_decoder_layers + ['cls']
    model = wrap_modules_data_parallel(model, dp_types_whole_model, dp_groups_whole_model, module_types=module_types, pp_devices=pp_devices_whole_model)

    # [Step 6] Construct Pipeline Parallel model
    chunks = get_chunks(args)
    model_hp = Pipe(model, chunks=chunks, checkpoint='never')
    return model_hp


def apply_layerwise_hybrid_strategy():
    ###### T5 Large 48 layers 1024 hidden
    ### A sample strategy
    # Note that nproc_per_node should be 8//pp_deg !
    pp_deg = 2
    tp_sizes_enc = [1]*48
    tp_consecutive_flags = [1]*48
    dp_types_enc = [0]*5+[1]*21+[0]*9+[1]*13

    if torch.distributed.get_rank() == 0:
        print('tp_sizes:', tp_sizes_enc)
        print('tp_consec:',tp_consecutive_flags)
        print('dp_type:',dp_types_enc)
        print('pp_deg:',pp_deg)

    return tp_sizes_enc, tp_consecutive_flags, dp_types_enc, pp_deg

# # T5 Large 32 pp divide
# pp_stage_dict_for_bsz = {8: {1: [32], 2: [16, 16], 4: [7, 11, 7, 7], 8: [6, 6, 5, 3, 3, 3, 3, 3]}, 
#                         16: {1: [32], 2: [17, 15], 4: [8, 10, 7, 7], 8: [6, 6, 5, 3, 3, 3, 3, 3]}, 
#                         24: {1: [32], 2: [18, 14], 4: [8, 10, 7, 7], 8: [6, 6, 5, 3, 3, 3, 3, 3]}, 
#                         32: {1: [32], 2: [18, 14], 4: [8, 10, 7, 7], 8: [6, 6, 5, 3, 3, 3, 3, 3]}}

# T5 Large 48 pp divide
pp_stage_dict_for_bsz = {8: {1: [48], 2: [26, 22], 4: [13, 15, 10, 10], 8: [6, 8, 8, 6, 5, 5, 5, 5]}, 
                        16: {1: [48], 2: [27, 21], 4: [14, 14, 10, 10], 8: [6, 8, 8, 6, 5, 5, 5, 5]}, 
                        24: {1: [48], 2: [27, 21], 4: [14, 14, 10, 10], 8: [7, 8, 8, 5, 5, 5, 5, 5]}}

def get_pp_ranks_enc(global_bsz, pp_deg):
    pp_ranks_enc = []
    pp_divide = pp_stage_dict_for_bsz[global_bsz][pp_deg]
    for i in range(pp_deg):
        pp_ranks_enc += [i]*pp_divide[i]
    if torch.distributed.get_rank() == 0:
        print(pp_divide)
        print(pp_ranks_enc)
    return pp_ranks_enc

def train(args):
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
    dataset = DataLoaderForT5(args)
    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input,
                            shuffle=False)

    print("Creating Model...")
    config = T5Config.from_pretrained(args.model_config, dropout_rate=args.dropout_prob)
    if args.num_encoder_layer > 0:
        config.num_layers = args.num_encoder_layer
    if args.num_decoder_layer > 0:
        config.num_decoder_layers = args.num_decoder_layer
    if args.num_head > 0:
        config.num_heads = args.num_head

    overwrite_megatron_args(config, args)
    if rank == 0:
        print(config.num_layers, config.num_decoder_layers, config.num_heads)

    t5_model = T5ForConditionalGeneration(config)

    if args.apply_strategy:
        tp_sizes_enc, tp_consecutive_flags, dp_types_enc, pp_deg = apply_layerwise_hybrid_strategy()
    else:
        assert(args.global_tp_deg > 0 and args.global_tp_consec in [0, 1])
        tp_sizes_enc = [args.global_tp_deg] * (config.num_layers+config.num_decoder_layers)
        tp_consecutive_flags = [args.global_tp_consec] * (config.num_layers+config.num_decoder_layers)
        dp_types_enc = [args.fsdp] * (config.num_layers+config.num_decoder_layers)

    # # divide pipeline stage averagely according to layer number, may cause imbalanced workload
    # avg_num_encoder_layers = config.num_layers // pp_deg
    # avg_num_decoder_layers = config.num_decoder_layers // pp_deg
    # pp_ranks_enc = []
    # if pp_deg == 1:
    #     pp_ranks_enc = [0]*(config.num_layers+config.num_decoder_layers)
    # else:
    #     pp_deg_enc = pp_deg_dec = pp_deg // 2
    #     avg_num_encoder_layers = config.num_layers // pp_deg_enc
    #     avg_num_decoder_layers = config.num_decoder_layers // pp_deg_dec
    #     for i in range(pp_deg_enc):
    #         pp_ranks_enc += [i] * avg_num_encoder_layers
    #     for i in range(pp_deg_dec):
    #         pp_ranks_enc += [i + pp_deg_enc] * avg_num_decoder_layers

    # divide pipeline stage averagely according to memory workload
    pp_ranks_enc = get_pp_ranks_enc(args.global_train_batch_size, pp_deg)

    model = construct_hybrid_parallel_model(t5_model, config, tp_sizes_enc, dp_types_enc, pp_ranks_enc, pp_deg, tp_consecutive_flags)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            input_ids, attention_mask, labels= [tensor.to(device) for tensor in batch]

            if args.profile and rank == profile_rank and iter <= 2:
                torch.cuda.reset_peak_memory_stats(rank)
                print_peak_memory("\nBefore Forward", rank, args.profile_type)

            # model forward
            logits = model(input_ids, labels, attention_mask).local_value()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

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
                print('[Epoch %d] (Iteration %d): Loss = %.3f, Time = %.3f'% (ep,iter,loss.item(), end_time-start_time))

def add_arguments(parser):
    group = parser.add_argument_group(title='our arguments')

    group.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    group.add_argument(
        "--global_train_batch_size", type=int, default=32, help="Training batch size"
    )
    group.add_argument(
        "--model_config", type=str, default='t5-base', help="T5 model name", choices=['t5-base', 't5-large']
    )
    group.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    group.add_argument(
        "-s", "--seq_length", type=int, default=512, help="Maximum sequence len"
    )
    group.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    group.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    group.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight_decay of adam"
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
        "--load_params", type=int, default=0, help="Whether to load saved init params."
    )
    parser.add_argument(
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8],
    )
    parser.add_argument(
        "--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8],
    )
    parser.add_argument(
        "--num_encoder_layer", type=int, default=0, help="overwrite encoder layer num"
    )
    parser.add_argument(
        "--num_decoder_layer", type=int, default=0, help="overwrite decoder layer num"
    )
    parser.add_argument(
        "--num_head", type=int, default=0, help="overwrite attention head num"
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
    args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'}
    initialize_megatron(extra_args_provider=add_arguments, args_defaults=args_defaults)
    args = get_args()
    set_seed()
    train(args)