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
    if local_bsz == 2:
        return 1
    elif strategy[0] == 8:
        return 3
    else:
        return 2

microbatch = True

# T5 Large 1024 Encoder config
parameter_size_enc = 48.01
forward_compute_time_per_layer = 35/24
tp_activation_per_bsz_dict_enc = {  1:91.003, 
                                2:54.004, 
                                4:32.785, 
                                8:22.8175}
memcost_model_args_enc = {  'parameter_size': parameter_size_enc,
                        'tp_activation_per_bsz_dict': tp_activation_per_bsz_dict_enc,
                        'other_model_states': 900,
                        'other_activation_per_bsz': 350}
timecost_model_args_with_overlap_enc = { 
                        'parameter_size': parameter_size_enc,
                        'microbatch': microbatch,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 512,
                        'hidden_size': 1024,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 60,
                        'comm_coe_dict': comm_coe_dict,
                        'dp_overlap_coe': 1.3,
                        'bct_overlap_coe': 1.3,
                        'layer_type': 'enc'}
timecost_model_args_without_overlap_enc = { 
                        'parameter_size': parameter_size_enc,
                        'microbatch': microbatch,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 512,
                        'hidden_size': 1024,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 60,
                        'comm_coe_dict': comm_coe_dict,
                        'layer_type': 'enc'}

# T5 Large 1024 Decoder config
parameter_size_dec = 64.012
forward_compute_time_per_layer = 65/24
tp_activation_per_bsz_dict_dec = {  1:157.755, 
                                2:90.756, 
                                4:56.384, 
                                8:39.117}
memcost_model_args_dec = {  'parameter_size': parameter_size_dec,
                        'tp_activation_per_bsz_dict': tp_activation_per_bsz_dict_dec,
                        'other_model_states': 900,
                        'other_activation_per_bsz': 350}
timecost_model_args_with_overlap_dec = { 
                        'parameter_size': parameter_size_dec,
                        'microbatch': microbatch,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 512,
                        'hidden_size': 1024,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 60,
                        'comm_coe_dict': comm_coe_dict,
                        'dp_overlap_coe': 1.3,
                        'bct_overlap_coe': 1.3,
                        'layer_type': 'dec'}
timecost_model_args_without_overlap_dec = { 
                        'parameter_size': parameter_size_dec,
                        'microbatch': microbatch,
                        'optimal_chunk_func': optimal_chunk_func,
                        'sequence_length': 512,
                        'hidden_size': 1024,
                        'forward_computation_time': forward_compute_time_per_layer,
                        'bct_fct_coe': 2,
                        'extra_overhead': 60,
                        'comm_coe_dict': comm_coe_dict,
                        'layer_type': 'dec'}

memcost_model_args=[memcost_model_args_enc, memcost_model_args_dec]
timecost_model_args=[timecost_model_args_with_overlap_enc, timecost_model_args_with_overlap_dec]

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
    bszs = list(range(8, 64, 8))
    pp_stage_dict_for_bsz = dict()
    for bsz in bszs:
        pp_stage_dict = dict()
        for pp_deg in [1,2,4,8]:
            pp_divide, mem_cost_per_stage = pp_stage_divide_greedy(memcost_model_args, [24, 24], pp_deg, bsz, strategies)
            #print(bsz, pp_deg, pp_divide, mem_cost_per_stage)
            pp_stage_dict[pp_deg] = pp_divide
        pp_stage_dict_for_bsz[bsz] = pp_stage_dict
    return pp_stage_dict_for_bsz


search_history = dict()
def search(max_mem):
    print("----Searching with max memory %d MB----"%max_mem)
    results = dict()
    max_throughput, optimal_bsz, max_bsz = -1, -1, -1
    for bsz in range(8, 256, 8):
        pp_stage_dict = pp_stage_dict_for_bsz[bsz]
        dp_on_model = DpOnModel(strategies, 
                                MemoryCostModel, 
                                TimeCostModel_with_overlap, 
                                memcost_model_args=memcost_model_args,
                                timecost_model_args=timecost_model_args,
                                max_mem=max_mem,
                                layer_num =[24, 24],
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
    bsz = 8
    layer_num = 24
    mem_0, mem_1, other = [], [], []
    for strategy in strategies:
        re = MemoryCostModel(strategy, global_batch_size=bsz, **memcost_model_args_enc).get_memory_cost()
        print(form_strategy(strategy), re['enc_total'], re['other'])
        mem_0.append(re['enc_total'])
        other.append(re['other'])
    print()
    for strategy in strategies:
        re = MemoryCostModel(strategy, global_batch_size=bsz, **memcost_model_args_dec).get_memory_cost()
        print(form_strategy(strategy), re['enc_total'], re['other'])
        mem_1.append(re['enc_total'])
    print()
    for i in range(len(strategies)):
        strategy = strategies[i]
        print(form_strategy(strategy), mem_0[i]*layer_num+mem_1[i]*layer_num+other[i])
    print()

    enc_re = []
    for strategy in strategies:
        re = TimeCostModel_with_overlap(strategy, global_batch_size=bsz, **timecost_model_args_with_overlap_enc).gen_result()
        print(form_strategy(strategy), re*layer_num)
        enc_re.append(re*layer_num)
    print()
    dec_re=[]
    for strategy in strategies:
        re = TimeCostModel_with_overlap(strategy, global_batch_size=bsz, **timecost_model_args_with_overlap_dec).gen_result()
        print(form_strategy(strategy), re*layer_num)
        dec_re.append(re*layer_num)
    print()
    for i in range(len(strategies)):
        print(form_strategy(strategies[i]), enc_re[i]+dec_re[i])
    print()

check_cost_model()
pp_stage_dict_for_bsz = get_pp_stages_for_all_bsz()
mem_list = [8, 12, 16, 20]
mem_list = [mem * 1024 for mem in mem_list]
for max_mem in mem_list:
    search(max_mem)
    print()