from enum import unique
import sys
sys.path.insert(0, '..')
from utils import MemoryCostModel, TimeCostModel_with_overlap
from utils import DpOnModel, print_strategies, form_strategy
import numpy as np

# full strategies
strategies = [[1,1,8,{'fsdp':0}],[1,1,8,{'fsdp':1}],
            [1,2,4,{'tp':0,'fsdp':0}],[1,2,4,{'tp':1,'fsdp':0}],[1,2,4,{'tp':0,'fsdp':1}],[1,2,4,{'tp':1,'fsdp':1}],
            [1,4,2,{'tp':0,'fsdp':0}],[1,4,2,{'tp':1,'fsdp':0}],[1,4,2,{'tp':0,'fsdp':1}],[1,4,2,{'tp':1,'fsdp':1}],
            [1,8,1,{}],
            [2,1,4,{'fsdp':0}],[2,1,4,{'fsdp':1}],
            [2,2,2,{'tp':0,'fsdp':0}],[2,2,2,{'tp':1,'fsdp':0}],[2,2,2,{'tp':0,'fsdp':1}],[2,2,2,{'tp':1,'fsdp':1}],
            [2,4,1,{}],
            [4,1,2,{'fsdp':0}],[4,1,2,{'fsdp':1}],
            [4,2,1,{}],
            [8,1,1,{}]]

# # only dp+tp
# strategies = [[1,1,8,{'fsdp':0}],
#             [1,2,4,{'tp':0,'fsdp':0}],[1,2,4,{'tp':1,'fsdp':0}],
#             [1,4,2,{'tp':0,'fsdp':0}],[1,4,2,{'tp':1,'fsdp':0}],
#             [1,8,1,{}]]

# # only dp+pp
# strategies = [[1,1,8,{'fsdp':0}],
#             [2,1,4,{'fsdp':0}],
#             [4,1,2,{'fsdp':0}],
#             [8,1,1,{}]]

# # only dp
# strategies = [[1,1,8,{'fsdp':0}]]

# # only tp
# strategies = [[1,8,1,{'fsdp':0}]]

# # only pp
# strategies = [[8,1,1,{'fsdp':0}]]

# # only fsdp
# strategies = [[1,1,8,{'fsdp':1}]]

comm_coe_dict={1:{'8':0.21530878, '4_0':0.220138889, '4_1':0.226519097, '2_0':0.271614583, '2_1':0.277994792, '1':0}, 
                2:{'4':0.230750868, '2_0':0.224153646, '2_1':0.275195313, '1':0}, 
                4:{'2':0.264908854, '1':0}, 
                8:{'1':0}}

def optimal_chunk_func(local_bsz, strategy):
    if local_bsz <= 8:
        return 1
    elif local_bsz > 8 and local_bsz < 32:
        return 2
    elif local_bsz >= 32 and local_bsz <= 96:
        return 3
    else:
        return 4

microbatch = True

# Swin Huge 1280 config for layer type 0
parameter_size = 4.80
forward_compute_time_per_layer = 1.25
tp_activation_per_bsz_dict = {  1:72.584, 
                                2:43.318, 
                                4:29.716, 
                                8:22.360}
memcost_model_args_layer_0 = {  'parameter_size': parameter_size,
                        'tp_activation_per_bsz_dict': tp_activation_per_bsz_dict,
                        'other_model_states': 312,
                        'other_activation_per_bsz': 70}
timecost_model_args_with_overlap_layer_0 = { 'parameter_size': parameter_size,
                        'microbatch': microbatch,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 49*64,
                        'hidden_size': 320,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 0,
                        'comm_coe_dict': comm_coe_dict,
                        'dp_overlap_coe': 1.3,
                        'bct_overlap_coe': 1.3}
timecost_model_args_without_overlap_layer_0 = { 'parameter_size': parameter_size,
                        'microbatch': microbatch,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 49*64,
                        'hidden_size': 320,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 0,
                        'comm_coe_dict': comm_coe_dict}

# Swin Huge 1280 config for layer type 1
parameter_size = 18.82
forward_compute_time_per_layer = 0.8875
tp_activation_per_bsz_dict = {  1:36.423, 
                                2:21.330, 
                                4:14.708, 
                                8:11.237}
memcost_model_args_layer_1 = {  'parameter_size': parameter_size,
                        'tp_activation_per_bsz_dict': tp_activation_per_bsz_dict,
                        'other_model_states': 312,
                        'other_activation_per_bsz': 70}
timecost_model_args_with_overlap_layer_1 = { 'parameter_size': parameter_size,
                        'microbatch': microbatch,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 49*16,
                        'hidden_size': 640,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 0,
                        'comm_coe_dict': comm_coe_dict,
                        'dp_overlap_coe': 1.3,
                        'bct_overlap_coe': 1.3}
timecost_model_args_without_overlap_layer_1 = { 'parameter_size': parameter_size,
                        'microbatch': microbatch,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 49*16,
                        'hidden_size': 640,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 0,
                        'comm_coe_dict': comm_coe_dict}

# Swin Huge 1280 config for layer type 2
parameter_size = 77.05
forward_compute_time_per_layer = 0.8
tp_activation_per_bsz_dict = {  1:18.317, 
                                2:11.252, 
                                4:7.561, 
                                8:5.611}
memcost_model_args_layer_2 = {  'parameter_size': parameter_size,
                        'tp_activation_per_bsz_dict': tp_activation_per_bsz_dict,
                        'other_model_states': 312,
                        'other_activation_per_bsz': 70}
timecost_model_args_with_overlap_layer_2 = { 'parameter_size': parameter_size,
                        'microbatch': microbatch,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 49*4,
                        'hidden_size': 1280,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 0,
                        'comm_coe_dict': comm_coe_dict,
                        'dp_overlap_coe': 1.3,
                        'bct_overlap_coe': 1.3}
timecost_model_args_without_overlap_layer_2 = { 'parameter_size': parameter_size,
                        'microbatch': microbatch,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 49*4,
                        'hidden_size': 1280,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 0,
                        'comm_coe_dict': comm_coe_dict}

# Swin Huge 1280 config for layer type 3
parameter_size = 302.22
forward_compute_time_per_layer = 1.225
tp_activation_per_bsz_dict = {  1:9.183, 
                                2:5.076, 
                                4:3.835, 
                                8:2.802}
memcost_model_args_layer_3 = {  'parameter_size': parameter_size,
                        'tp_activation_per_bsz_dict': tp_activation_per_bsz_dict,
                        'other_model_states': 312,
                        'other_activation_per_bsz': 70}
timecost_model_args_with_overlap_layer_3 = { 'parameter_size': parameter_size,
                        'microbatch': microbatch,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 49*1,
                        'hidden_size': 2560,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 0,
                        'comm_coe_dict': comm_coe_dict,
                        'dp_overlap_coe': 1.3,
                        'bct_overlap_coe': 1.3}
timecost_model_args_without_overlap_layer_3 = { 'parameter_size': parameter_size,
                        'microbatch': microbatch,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 49*1,
                        'hidden_size': 2560,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 0,
                        'comm_coe_dict': comm_coe_dict}


memcost_model_args = [memcost_model_args_layer_0, memcost_model_args_layer_1, memcost_model_args_layer_2, memcost_model_args_layer_3]
timecost_model_args = [timecost_model_args_with_overlap_layer_0, timecost_model_args_with_overlap_layer_1, timecost_model_args_with_overlap_layer_2, timecost_model_args_with_overlap_layer_3]

def pp_stage_divide_greedy(memcost_model_args, layer_num, pp_deg, bsz, strategies):
    assert(len(memcost_model_args)==len(layer_num))
    if pp_deg == 1:
        return [np.sum(layer_num)], None
    layer_type_num = len(layer_num)
    layer_min_memcost = []
    strategies = list(filter(lambda s: s[0] == pp_deg, strategies))
    if len(strategies)==0:
        return None, None
    for i in range(layer_type_num):
        memcosts = [MemoryCostModel(strategy, global_batch_size=bsz, **memcost_model_args[i]).get_memory_cost()['enc_total'] for strategy in strategies]
        layer_min_memcost.append(np.min(memcosts))
    other_cost = MemoryCostModel([1,1,8,{'fsdp':0}], global_batch_size=bsz, **memcost_model_args[0]).get_memory_cost()['other']
    #print(layer_min_memcost, other_cost)
    min_memcost_all_layers = []
    for i in range(layer_type_num):
        min_memcost_all_layers += [layer_min_memcost[i]]*layer_num[i]
    #print(min_memcost_all_layers)
    avg_mem_cost = (np.sum(min_memcost_all_layers)+other_cost)/pp_deg
    #print('Avg memcost:', avg_mem_cost)

    pp_divide = [0]*pp_deg
    mem_cost_per_stage = [other_cost] + [0] * (pp_deg-1)
    idx = len(min_memcost_all_layers)-1
    for i in range(pp_deg-1,-1,-1):
        while True:
            if idx < 0:
                break
            if i > 0 and avg_mem_cost - mem_cost_per_stage[i] < 0.5 * min_memcost_all_layers[idx]:
                break
            else:
                mem_cost_per_stage[i]+=min_memcost_all_layers[idx]
                idx-=1
                pp_divide[i]+=1
    if pp_divide[0] == 0:
        pp_divide[0] += 1
        pp_divide[1] -= 1
        mem_cost_per_stage[0] += min_memcost_all_layers[0]
        mem_cost_per_stage[1] -= min_memcost_all_layers[0]
    return pp_divide, mem_cost_per_stage

def get_pp_stages_for_all_bsz():
    bszs = list(range(8, 256, 8))
    pp_stage_dict_for_bsz = dict()
    for bsz in bszs:
        pp_stage_dict = dict()
        for pp_deg in [1,2,4,8]:
            pp_divide, mem_cost_per_stage = pp_stage_divide_greedy(memcost_model_args, [2, 2, 26, 2], pp_deg, bsz, strategies)
            # print(bsz, pp_deg, pp_divide, mem_cost_per_stage)
            pp_stage_dict[pp_deg] = pp_divide
        pp_stage_dict_for_bsz[bsz] = pp_stage_dict
    return pp_stage_dict_for_bsz

search_history = dict()
def search(max_mem):
    print("----Searching with max memory %d MB----"%max_mem)
    results = dict()
    max_throughput, optimal_bsz, max_bsz = -1, -1, -1
    for bsz in range(8, 1024, 8):
        pp_stage_dict = pp_stage_dict_for_bsz[bsz]
        dp_on_model = DpOnModel(strategies, 
                                MemoryCostModel, 
                                TimeCostModel_with_overlap, 
                                memcost_model_args=memcost_model_args,
                                timecost_model_args=timecost_model_args,
                                max_mem=max_mem,
                                layer_num =[2, 2, 26, 2],
                                multi_layer_type = True,
                                pp_stage_dict = pp_stage_dict,
                                search_history=search_history,
                                comm_coe_dict=comm_coe_dict)
        print("****Testing with bsz=", bsz, "****")
        min_cost, min_res_list, min_pp_deg, mem_remain, mem_cost = dp_on_model.fit(bsz)
        throughput = bsz / min_cost
        print(f"[Optimal pp_deg={min_pp_deg}] Minimized timecost={min_cost} Memory remaining={mem_remain} Memory cost={mem_cost}")
        print(f"Max throughput={throughput} samples/s")
        print_strategies(min_res_list)
        results[bsz] = {'min_cost': min_cost, 'min_res_list': min_res_list, 'min_pp_deg': min_pp_deg, 
                        'mem_remain': mem_remain, 'mem_cost': mem_cost, 'throughput': throughput}
        if throughput > max_throughput:
            max_throughput = throughput
            optimal_bsz = bsz
        if min_pp_deg == -1:
            break
        max_bsz = bsz

    print('\nFinal results of max memory %d MB:'%max_mem)
    re = results[optimal_bsz]
    print(f"Optimal bsz = {optimal_bsz} Max throughput={re['throughput']} samples/s")
    print(f"pp_deg={re['min_pp_deg']} Minimized timecost={re['min_cost']} Memory remaining={re['mem_remain']} Memory cost={re['mem_cost']}")
    print_strategies(re['min_res_list'])
    if max_bsz > -1 and max_bsz != optimal_bsz:
        re = results[max_bsz]
        print(f"\nMax bsz = {max_bsz} Max throughput={re['throughput']} samples/s")
        print(f"pp_deg={re['min_pp_deg']} Minimized timecost={re['min_cost']} Memory remaining={re['mem_remain']} Memory cost={re['mem_cost']}")
        print_strategies(re['min_res_list'])
    print("-----------------------------------------")

# Check cost model
def check_cost_model():
    bsz=32
    layer_num=18
    mem_0, mem_1, mem_2, mem_3, other = [], [], [], [], []
    for strategy in strategies:
        re = MemoryCostModel(strategy, global_batch_size=bsz, **memcost_model_args_layer_0).get_memory_cost()
        print(form_strategy(strategy), re['enc_total'], re['other'])
        mem_0.append(re['enc_total'])
        other.append(re['other'])
    print()
    for strategy in strategies:
        re = MemoryCostModel(strategy, global_batch_size=bsz, **memcost_model_args_layer_1).get_memory_cost()
        print(form_strategy(strategy), re['enc_total'], re['other'])
        mem_1.append(re['enc_total'])
    print()
    for strategy in strategies:
        re = MemoryCostModel(strategy, global_batch_size=bsz, **memcost_model_args_layer_2).get_memory_cost()
        print(form_strategy(strategy), re['enc_total'], re['other'])
        mem_2.append(re['enc_total'])
    print()
    for strategy in strategies:
        re = MemoryCostModel(strategy, global_batch_size=bsz, **memcost_model_args_layer_3).get_memory_cost()
        print(form_strategy(strategy), re['enc_total'], re['other'])
        mem_3.append(re['enc_total'])
    print()
    for i in range(len(strategies)):
        strategy = strategies[i]
        print(form_strategy(strategy), mem_0[i]*2+mem_1[i]*2+mem_2[i]*layer_num+mem_3[i]*2+other[i])
    print()

    time_0, time_1, time_2, time_3 = [], [], [], []
    for strategy in strategies:
        re = TimeCostModel_with_overlap(strategy, global_batch_size=bsz, **timecost_model_args_with_overlap_layer_0).gen_result()
        print(form_strategy(strategy), re*2)
        time_0.append(re)
    print()
    for strategy in strategies:
        re = TimeCostModel_with_overlap(strategy, global_batch_size=bsz, **timecost_model_args_with_overlap_layer_1).gen_result()
        print(form_strategy(strategy), re*2)
        time_1.append(re)
    print()
    for strategy in strategies:
        re = TimeCostModel_with_overlap(strategy, global_batch_size=bsz, **timecost_model_args_with_overlap_layer_2).gen_result()
        print(form_strategy(strategy), re*8)
        time_2.append(re)
    print()
    for strategy in strategies:
        re = TimeCostModel_with_overlap(strategy, global_batch_size=bsz, **timecost_model_args_with_overlap_layer_3).gen_result()
        print(form_strategy(strategy), re*2)
        time_3.append(re)
    print()

    for i in range(len(strategies)):
        strategy = strategies[i]
        print(form_strategy(strategy), time_0[i]*2+time_1[i]*2+time_2[i]*layer_num+time_3[i]*2)


check_cost_model()
pp_stage_dict_for_bsz = get_pp_stages_for_all_bsz()
# print(pp_stage_dict_for_bsz)
mem_list = [8, 12, 16, 20]
mem_list = [mem * 1024 for mem in mem_list]
for max_mem in mem_list:
    search(max_mem)
    print()