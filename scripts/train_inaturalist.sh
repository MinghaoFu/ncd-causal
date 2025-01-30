#!/bin/bash
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk -v gpu_id=$CUDA_VISIBLE_DEVICES 'NR==gpu_id+1 {print "GPU-Util:", $1}'
data_set=Fungi
seed=1026
output_dir=exp/
run_name="${daset_set}_seed($seed)_zs3072"
#run_name=test
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F ',' '{if ($1 >= 0 && $1 <= 3) print $0}' | sort -t ',' -k2 -n | head -n 1 | awk -F ',' '{print $1}') python main.py \
    --data_set=$data_set \
    --output_dir=$output_dir/$data_set/${run_name} \
    --seed=$seed \
    --num_workers 20 \
    --l_recon 1 \
    --l_spa 0 \
    --l_intra 1e-2 \
    --epochs 400 \
    --n_intra 1 \
    --temperature 1 \
    --n_recon_epoch 200 \
    --hash_code_length 32 \
    --l_ind 1e-2 \
    --zc_dim 0 \
    --zs_dim 3072 \
    # --eval \
    # --resume "/home/xinyu.li/minghao.fu/xinyu-ncd/exp/cub/cub_seed(1026)_zs3072_sparsity5e-2/checkpoints/best_model.pth"
    #--resume "/home/xinyu.li/minghao.fu/xinyu-ncd/exp/cub/cub_seed(1026)_recon1/checkpoints/best_model.pth" 
    #--resume "/home/xinyu.li/minghao.fu/xinyu-ncd/exp/cub/cub_seed(1026)_pretrained_from_recon1_synthetic/checkpoints/best_model.pth" 
    