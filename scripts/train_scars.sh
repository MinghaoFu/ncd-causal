#!/bin/bash
data_set=scars
seed=1027
output_dir=exp/
run_name="scars_seed($seed)_zs2048_nopatchrecon_spa5e-2"

CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F ',' '{if ($1 >= 0 && $1 <= 3) print $0}' | sort -t ',' -k2 -n | head -n 1 | awk -F ',' '{print $1}') python main.py \
    --data_set=$data_set \
    --output_dir=$output_dir/$data_set/${run_name} \
    --seed=$seed \
    --num_workers 20 \
    --l_recon 1 \
    --l_spa 5e-2 \
    --l_intra 1e-2 \
    --epochs 400 \
    --n_intra 1 \
    --temperature 1 \
    --n_recon_epoch 200 \
    --hash_code_length 32 \
    --l_ind 1e-2 \
    --zc_dim 0 \
    --zs_dim 2048 \