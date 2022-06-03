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
    if strategy[0] == 1:
        return 1
    if np.array_equal(strategy, [2,4,1]) or np.array_equal(strategy, [4,2,1]):
        return local_bsz/4
    if local_bsz < 6:
        return local_bsz/2
    else:
        return local_bsz/8+2

# Bert Huge 1280 config
parameter_size = 76.98
forward_compute_time_per_layer = 55 / 24
tp_activation_per_bsz_dict = {  1:94.07, 
                                2:52.475, 
                                4:31.754375, 
                                8:21.61171875}
memcost_model_args = {  'parameter_size': parameter_size,
                        'tp_activation_per_bsz_dict': tp_activation_per_bsz_dict,
                        'other_model_states': 793,
                        'other_activation_per_bsz': 313}
timecost_model_args_with_overlap = { 
                        'parameter_size': parameter_size,
                        'microbatch': True,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 512,
                        'hidden_size': 1280,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 100,
                        'comm_coe_dict': comm_coe_dict,
                        'dp_overlap_coe': 1.3,
                        'bct_overlap_coe': 1.3}
timecost_model_args_without_overlap = { 
                        'parameter_size': parameter_size,
                        'microbatch': True,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 512,
                        'hidden_size': 1280,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 100,
                        'comm_coe_dict': comm_coe_dict}

search_history = dict()
def search(max_mem):
    print("----Searching with max memory %d MB----"%max_mem)
    dp_on_model = DpOnModel(strategies, 
                            MemoryCostModel, 
                            TimeCostModel_with_overlap, 
                            memcost_model_args,
                            timecost_model_args_with_overlap,
                            max_mem=max_mem,
                            search_history=search_history,
                            layer_num=32,
                            comm_coe_dict=comm_coe_dict)

    results = dict()
    max_throughput, optimal_bsz, max_bsz = -1, -1, -1
    for bsz in range(8, 256, 8):
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
    bsz = 8
    for strategy in strategies:
        re = MemoryCostModel(strategy, global_batch_size=bsz, **memcost_model_args).get_memory_cost()
        print(form_strategy(strategy), re['enc_total'], re['other'], re['enc_total']*24/strategy[0]+re['other']-1024)
    print()
    for strategy in strategies:
        re = TimeCostModel_with_overlap(strategy, global_batch_size=bsz, **timecost_model_args_with_overlap).gen_result()
        print(form_strategy(strategy), re*24)
    print()

check_cost_model()
mem_list = [8, 12, 16, 20]
mem_list = [mem * 1024 for mem in mem_list]
for max_mem in mem_list:
    search(max_mem)
    print()