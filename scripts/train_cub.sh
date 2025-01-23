#!/bin/bash
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk -v gpu_id=$CUDA_VISIBLE_DEVICES 'NR==gpu_id+1 {print "GPU-Util:", $1}'
data_set=cub
seed=1026
output_dir=exp/
#run_name="cub_seed($seed)_syn_iter2old"
run_name=test
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t ',' -k2 -n | head -n 1 | awk -F ',' '{print $1}') python main.py \
    --data_set=$data_set \
    --output_dir=$output_dir/$data_set/${run_name} \
    --seed=$seed \
    --num_workers 20 \
    --l_recon 1 \
    --l_spa 1e-2 \
    --l_intra 1e-2 \
    --epochs 400 \
    --n_intra 1 \
    --temperature 1 \
    --n_recon_epoch 200 \
    --hash_code_length 32 \
    --l_ind 1e-2 \
    --zc_dim 0 \
    --syn \
    --resume "/home/xinyu.li/minghao.fu/xinyu-ncd/exp/cub/cub_seed(1026)_pretrained_from_recon1_synthetic/checkpoints/best_model.pth" 
    #--resume "/home/xinyu.li/minghao.fu/xinyu-ncd/exp/cub/cub_seed(1026)_recon1/checkpoints/best_model.pth" 