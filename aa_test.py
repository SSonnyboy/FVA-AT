import argparse
import json
import os
import torch
from pathlib import Path

from autoattack import AutoAttack
from utils.tools import set_seed
from datasets.idx import *
from models import get_model


def load_config_from_json(config_path):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    class Config:
        pass

    config = Config()
    for key, value in config_dict.items():
        setattr(config, key, value)

    return config


def main():
    parser = argparse.ArgumentParser(description='AutoAttack Test for Adversarial Trained Model')

    parser.add_argument('--model_dir', type=str, default='./outputs',
                        help='Path to model directory containing experiment folders')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name. If None, test all experiments.')
    parser.add_argument('--model_type', type=str, default='best_pgd_model.pth',
                        choices=['best_pgd_model.pth', 'latest_model.pth'],
                        help='Which model to test')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for AutoAttack evaluation')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Epsilon for attack (if None, use config.epsilon)')

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(args.gpu_id)

    if args.exp_name:
        exp_dirs = [os.path.join(args.model_dir, args.exp_name)]
    else:
        exp_dirs = [os.path.join(args.model_dir, d) for d in os.listdir(args.model_dir)
                    if os.path.isdir(os.path.join(args.model_dir, d))]

    for exp_dir in sorted(exp_dirs):
        if not os.path.exists(exp_dir):
            print(f"Experiment directory not found: {exp_dir}")
            continue

        config_path = os.path.join(exp_dir, 'config.json')
        model_path = os.path.join(exp_dir, args.model_type)

        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            continue

        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue

        exp_name = os.path.basename(exp_dir)
        model_suffix = 'best' if 'best' in args.model_type else 'latest'
        log_path = os.path.join(exp_dir, f'{exp_name}_{model_suffix}_aa_test.log')

        if os.path.exists(log_path):
            print(f"AutoAttack results already exist: {log_path}, skipping...")
            continue

        print(f"\n{'='*70}")
        print(f"AutoAttack Testing: {exp_name}")
        print(f"{'='*70}")

        config = load_config_from_json(config_path)

        set_seed(config.seed)

        if config.dataset == "cifar10":
            _, test_loader, _ = get_cifar10_loaders_idx(config)
            config.num_classes = 10
        elif config.dataset == "cifar100":
            _, test_loader, _ = get_cifar100_loaders_idx(config)
            config.num_classes = 100
        elif config.dataset == "svhn":
            _, test_loader, _ = get_svhn_loaders_idx(config)
            config.num_classes = 10
        else:
            print(f"Unknown dataset: {config.dataset}")
            continue

        X_all, y_all = [], []
        for images, labels, _ in test_loader:
            X_all.append(images)
            y_all.append(labels)

        X = torch.cat(X_all).to(device)
        y = torch.cat(y_all).to(device)

        epsilon = args.epsilon if args.epsilon is not None else config.epsilon

        print(f"Dataset: {config.dataset}")
        print(f"Model: {config.model}")
        print(f"Mode: {config.mode}")
        print(f"Epsilon: {epsilon}")
        print(f"Test samples: {len(y)}")
        print(f"Log path: {log_path}\n")

        model = get_model(config).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        attacker = AutoAttack(
            model,
            norm='Linf',
            eps=epsilon / 255,
            device=device,
            version='standard',
            log_path=log_path
        )

        attacker.run_standard_evaluation(X, y, bs=args.batch_size)

        print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
