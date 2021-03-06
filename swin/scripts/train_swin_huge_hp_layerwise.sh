python -m torch.distributed.launch --nproc_per_node=8 --master_port 9995 train_hp_layerwise.py \
--seq-length 320 \
--global_train_batch_size 64 \
--embed_dim 320 \
--depths 2 2 26 2 \
--num_heads 8 16 32 64 \
--window_size 7 \
--epochs 10 \
--lr 1.25e-4 \
--adam_weight_decay 0.05 \
--data-folder ImageNet \
--pp_deg 1 \
--global_tp_deg 1 \
--global_tp_consec 1 \
--chunks 1 \
--fsdp 0 \
--profile 0 \
--apply_strategy 0 \
--check_loss 0