#!/bin/bash
data_set=pets
seed=1029
output_dir=exp/
known_scale=0.8
run_name="pets_seed($seed)_zs3072_zc32_${known_scale}"
#run_name=test

CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t ',' -k2 -n | head -n 1 | awk -F ',' '{print $1}') python main.py \
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
    --zc_dim 32 \
    --zs_dim 3072 \
    --prop_train_labels ${known_scale} \