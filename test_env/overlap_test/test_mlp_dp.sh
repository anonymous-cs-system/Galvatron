python -m torch.distributed.launch --nproc_per_node=8 --master_port 9999 test.py \
--train_batch_size 32 \
--ddp_on 1