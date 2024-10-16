
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 run_pretrain.py --batch_size 64 --model MIM_vit_base_patch16 \
--hog_nbins 9 --mask_ratio 0.75 \
--epochs 400 --warmup_epochs 10 --blr 1e-3 --weight_decay 0.05 \
--data_path "../data/reflacx-1.0.0/" --output_dir "./output_dir/"