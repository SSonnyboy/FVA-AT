#!/bin/bash
mkdir -p logs
nohup python src/train.py \
    --dataset cifar10 \
    --model resnet18 \
    --config pgd_at \
    --seed 42 \
    --gpu_id 1 \
    --out_dir outputs/ \
    > logs/cifar10.out 2>&1 & 
# python src/train.py \
#     --dataset cifar10 \
#     --model preactresnet18 \
#     --config pgd_at \
#     --seed 1 \
#     --gpu_id 1 \
#     --out_dir outputs/