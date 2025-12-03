import argparse
import json
import os
import csv
import torch
import torch.nn as nn
from pathlib import Path

from attacks import *
from utils.tools import set_seed
from losses import *
from datasets.idx import *
from models import get_model
from utils.extractor import *
from utils.rescore import rescore
from torchattacks import PGD as PGD_eval
from utils.cw import CW_Linf


def evaluate_model(model, test_loader, device, epsilon, alpha, n_steps, dataset_type='cifar10'):
    model.eval()

    clean_correct = 0
    pgd10_correct = 0
    pgd20_correct = 0
    pgd50_correct = 0
    cw_correct = 0
    
    total = 0

    attacker_pgd10 = PGD_eval(model, eps=epsilon/255, alpha=alpha/255, steps=10)
    attacker_pgd20 = PGD_eval(model, eps=epsilon/255, alpha=alpha/255, steps=20)
    attacker_pgd50 = PGD_eval(model, eps=epsilon/255, alpha=alpha/255, steps=50)
    attacker_cw = CW_Linf(model, eps=epsilon/255, alpha=alpha/255, steps=50)

    with torch.no_grad():
        for images, labels, index in test_loader:
            images, labels = images.to(device), labels.to(device)

            logits_clean = model(images)
            clean_pred = logits_clean.argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()
            with torch.enable_grad():
                img_adv_pgd10 = attacker_pgd10(images, labels)
                logits_pgd10 = model(img_adv_pgd10)
                pgd10_pred = logits_pgd10.argmax(dim=1)
                pgd10_correct += (pgd10_pred == labels).sum().item()

                img_adv_pgd20 = attacker_pgd20(images, labels)
                logits_pgd20 = model(img_adv_pgd20)
                pgd20_pred = logits_pgd20.argmax(dim=1)
                pgd20_correct += (pgd20_pred == labels).sum().item()

                img_adv_pgd50 = attacker_pgd50(images, labels)
                logits_pgd50 = model(img_adv_pgd50)
                pgd50_pred = logits_pgd50.argmax(dim=1)
                pgd50_correct += (pgd50_pred == labels).sum().item()

                img_adv_cw = attacker_cw(images, labels)
                logits_cw = model(img_adv_cw)
                cw_pred = logits_cw.argmax(dim=1)
                cw_correct += (cw_pred == labels).sum().item()

            total += labels.size(0)

    clean_acc = 100.0 * clean_correct / total
    pgd10_acc = 100.0 * pgd10_correct / total
    pgd20_acc = 100.0 * pgd20_correct / total
    pgd50_acc = 100.0 * pgd50_correct / total
    cw_acc = 100.0 * cw_correct / total
    

    return {
        'Clean_Acc': clean_acc,
        'PGD10_Acc': pgd10_acc,
        'PGD20_Acc': pgd20_acc,
        'PGD50_Acc': pgd50_acc,
        'CW50_Acc': cw_acc,
        
    }


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
    parser = argparse.ArgumentParser(description='Test Adversarial Trained Model')

    parser.add_argument('--model_dir', type=str, default='./outputs',
                        help='Path to model directory containing experiment folders')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (e.g., cifar10_resnet18_None_at). If None, test all experiments.')
    parser.add_argument('--model_type', type=str, default='best_pgd_model.pth',
                        choices=['best_pgd_model.pth', 'latest_model.pth'],
                        help='Which model to test')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='GPU ID to use')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for testing')

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

        print(f"\n{'='*60}")
        print(f"Testing experiment: {os.path.basename(exp_dir)}")
        print(f"{'='*60}")

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

        model = get_model(config).to(device)
        model.load_state_dict(torch.load(model_path))

        print(f"Dataset: {config.dataset}")
        print(f"Model: {config.model}")
        print(f"Mode: {config.mode}")
        print(f"Epsilon: {config.epsilon}")
        print(f"Testing {config.batch_size}...")

        results = evaluate_model(
            model,
            test_loader,
            device,
            epsilon=config.epsilon,
            alpha=config.alpha,
            n_steps=config.n_steps,
            dataset_type=config.dataset
        )
        if "best" in args.model_type:
            results_csv_path = os.path.join(exp_dir, 'test_results_best.csv')
        else:
            results_csv_path = os.path.join(exp_dir, 'test_results_last.csv')
            

        with open(results_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Accuracy (%)'])
            for metric, acc in results.items():
                writer.writerow([metric, f'{acc:.2f}'])

        print(f"\nTest Results:")
        print(f"  Clean Accuracy:  {results['Clean_Acc']:.2f}%")
        print(f"  PGD-10 Accuracy: {results['PGD10_Acc']:.2f}%")
        print(f"  PGD-20 Accuracy: {results['PGD20_Acc']:.2f}%")
        print(f"  PGD-50 Accuracy: {results['PGD50_Acc']:.2f}%")
        print(f"  CW-50 Accuracy: {results['CW50_Acc']:.2f}%")
        
        print(f"\nResults saved to: {results_csv_path}")


if __name__ == "__main__":
    main()
