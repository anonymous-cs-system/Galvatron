nvprof --profile-child-processes -o ./nvvps/comm_coe_8_%p.nvvp python -m torch.distributed.launch --nproc_per_node=8 --master_port 9996 test.py \
--global_tp_deg 8 \
--global_tp_consec 1 \
--pp_deg 1

nvprof --profile-child-processes -o ./nvvps/comm_coe_4_0_%p.nvvp python -m torch.distributed.launch --nproc_per_node=8 --master_port 9996 test.py \
--global_tp_deg 4 \
--global_tp_consec 0 \
--pp_deg 1

nvprof --profile-child-processes -o ./nvvps/comm_coe_4_1_%p.nvvp python -m torch.distributed.launch --nproc_per_node=8 --master_port 9996 test.py \
--global_tp_deg 4 \
--global_tp_consec 1 \
--pp_deg 1

nvprof --profile-child-processes -o ./nvvps/comm_coe_2_0_%p.nvvp python -m torch.distributed.launch --nproc_per_node=8 --master_port 9996 test.py \
--global_tp_deg 2 \
--global_tp_consec 0 \
--pp_deg 1

nvprof --profile-child-processes -o ./nvvps/comm_coe_2_1_%p.nvvp python -m torch.distributed.launch --nproc_per_node=8 --master_port 9996 test.py \
--global_tp_deg 2 \
--global_tp_consec 1 \
--pp_deg 1







