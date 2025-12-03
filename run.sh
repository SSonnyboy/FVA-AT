mkdir -p logs

nohup python train.py \
    --save_imd \
    --dataset cifar10 \
    --model resnet18 \
    --perturbation awp \
    --mode at \
    --gpu_id 1 \
    > logs/cifar10.out 2>&1 & 
