import os
import sys
import torch
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from common.config import Config
from datasets import get_loaders
from models import get_model
from evals import (
    evaluate_natural, evaluate_pgd_10, evaluate_pgd_20, evaluate_pgd_50,
    evaluate_cw, evaluate_aa
)
from utils import load_checkpoint


def find_experiment_folder(exp_path):
    """
    查找实验文件夹，支持新的两级结构:
    outputs/{dataset}_{model}_{method}_{perturbation}/seed_{seed}_{timestamp}/

    可以接受:
    1. 完整路径
    2. 子文件夹名 (seed_xxx_yyyymmdd_hhmmss)
    3. 主文件夹名 (dataset_model_method_perturbation) - 返回该文件夹下最新的子文件夹
    """
    exp_path = Path(exp_path)

    # 如果是完整路径且存在
    if exp_path.exists() and exp_path.is_dir():
        return exp_path

    outputs_dir = Path(__file__).parent.parent / 'outputs'

    # 搜索方式1: 在所有主文件夹中查找匹配的子文件夹
    for main_dir in outputs_dir.glob('*'):
        if main_dir.is_dir():
            for sub_dir in main_dir.glob('*'):
                if sub_dir.is_dir() and (sub_dir.name == exp_path.name or exp_path.name in sub_dir.name):
                    return sub_dir

    # 搜索方式2: 如果输入是主文件夹名，返回其下最新的子文件夹
    main_dir = outputs_dir / exp_path.name
    if main_dir.exists() and main_dir.is_dir():
        sub_dirs = sorted([d for d in main_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
        if sub_dirs:
            print(f"⚠ Found main experiment folder: {main_dir}")
            print(f"  Using latest sub-folder: {sub_dirs[0].name}")
            return sub_dirs[0]

    raise FileNotFoundError(f"Cannot find experiment folder: {exp_path}")


def find_checkpoint(exp_folder, checkpoint_type='best'):
    exp_folder = Path(exp_folder)
    checkpoints_dir = exp_folder / 'checkpoints'

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    if checkpoint_type == 'best':
        checkpoint_file = checkpoints_dir / 'best_model.pt'
    elif checkpoint_type == 'final':
        checkpoint_file = checkpoints_dir / 'final_model.pt'
    else:
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    return checkpoint_file


def load_exp_config(exp_folder):
    """
    从实验文件夹路径中加载配置

    支持两种文件夹结构:
    1. 新结构（推荐）: outputs/{dataset}_{model}_{method}_{perturbation}/seed_{seed}_{YYYYMMDD_HHMMSS}/
    2. 旧结构: outputs/{dataset}_{model}_{method}_{seed}_{perturbation}_{YYYYMMDD_HHMMSS}/
    """
    exp_folder = Path(exp_folder)
    config = Config()

    # 已知的模型和 perturbation 列表
    MODELS = ['resnet18', 'wrn28_10', 'wrn34_10', 'preactresnet18']
    PERTURBATIONS = ['none', 'awp', 'rwp', 'uawp']
    DATASETS = ['cifar10', 'cifar100', 'svhn']

    # 检查是否为新结构：主文件夹存在且不以时间戳结尾
    parent_dir = exp_folder.parent
    sub_folder_name = exp_folder.name
    main_folder_name = parent_dir.name if parent_dir.name != 'outputs' else None

    # 尝试从主文件夹名和子文件夹名解析配置
    if main_folder_name and sub_folder_name.startswith('seed_'):
        # 新结构：sub_folder_name = "seed_{seed}_{YYYYMMDD_HHMMSS}"
        try:
            # 解析子文件夹名
            sub_parts = sub_folder_name.split('_')
            # seed_{seed}_{YYYYMMDD}_{HHMMSS}
            if len(sub_parts) >= 4 and sub_parts[0] == 'seed':
                seed = int(sub_parts[1])
                config['seed'] = seed

            # 解析主文件夹名: {dataset}_{model}_{method}_{perturbation}
            main_parts = main_folder_name.split('_')

            if len(main_parts) >= 2:
                # 第一部分是 dataset
                dataset = main_parts[0]
                config['dataset'] = dataset

                # 最后一部分是 perturbation
                perturbation = main_parts[-1]
                if perturbation in PERTURBATIONS:
                    config['perturbation'] = perturbation
                else:
                    # 如果不在已知列表中，可能是 method 的一部分
                    perturbation = None

                # 中间部分包含 model 和 method
                remaining_str = '_'.join(main_parts[1:] if not perturbation else main_parts[1:-1])

                # 尝试找到 model
                model = None
                for m in MODELS:
                    if remaining_str.startswith(m):
                        model = m
                        method = remaining_str[len(m):].lstrip('_') if len(remaining_str) > len(m) else ''
                        break

                if model:
                    config['model'] = model
                    config['method'] = method if method else 'unknown'
                    if perturbation:
                        config['perturbation'] = perturbation
                    return _complete_config(config)

        except (ValueError, IndexError):
            pass

    # 降级到旧结构解析
    folder_name = exp_folder.name
    parts = folder_name.split('_')

    # 旧格式: {dataset}_{model}_{method}_{seed}_{perturbation}_{YYYYMMDD}_{HHMMSS}
    if len(parts) >= 8:
        # 移除时间戳（最后两部分）
        remaining_parts = parts[:-2]

        # 第一部分是 dataset
        dataset = remaining_parts[0]
        config['dataset'] = dataset

        # 倒数最后一部分是 perturbation
        perturbation = remaining_parts[-1]

        # 倒数第二部分应该是 seed (数字)
        try:
            seed = int(remaining_parts[-2])

            # 尝试找到 model
            model = None
            for m in MODELS:
                remaining_str = '_'.join(remaining_parts[1:-2])
                if remaining_str.startswith(m):
                    model = m
                    method_str = remaining_str[len(m):].lstrip('_')
                    method = method_str if method_str else 'unknown'
                    break

            if model is None:
                model = remaining_parts[1] if len(remaining_parts) > 2 else 'unknown'
                method = 'unknown'

            config['model'] = model
            config['method'] = method
            config['seed'] = seed
            config['perturbation'] = perturbation
            return _complete_config(config)

        except (ValueError, IndexError):
            # 如果无法解析，设置基本信息
            config['dataset'] = remaining_parts[0]
            config['model'] = remaining_parts[1] if len(remaining_parts) > 1 else 'unknown'
            return _complete_config(config)

    return _complete_config(config)


def _complete_config(config):
    """补完配置的其他必要字段"""
    config['data_dir'] = f'./data/{config.get("dataset", "cifar10").upper()}'
    config['num_classes'] = 100 if config.get("dataset") == 'cifar100' else 10
    config['batch_size'] = 128
    config['num_workers'] = 4
    config['pin_memory'] = True
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate model from experiment folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (New structure: outputs/{dataset}_{model}_{method}_{perturbation}/seed_{seed}_{timestamp}/):
  # Evaluate best PGD-10 model (default)
  python eval.py --exp seed_42_20240101_120000

  # Evaluate final model with full path
  python eval.py --exp ../outputs/cifar10_resnet18_pgd_at_none/seed_42_20240101_120000 --ckpt final

  # Using main folder name (auto-selects latest sub-folder)
  python eval.py --exp cifar10_resnet18_pgd_at_none --gpu 0

  # Specify GPU and run all attacks
  python eval.py --exp seed_42_20240101_120000 --gpu 0 --all-attacks

  # Do not save results
  python eval.py --exp seed_42_20240101_120000 --no-save
        """
    )

    # 实验相关参数
    parser.add_argument('--exp', type=str, required=True,
                        help='Experiment folder name or path')
    parser.add_argument('--ckpt', type=str, default='best',
                        choices=['best', 'final'],
                        help='Which checkpoint to evaluate: "best" (best PGD-10 acc), "final" (last epoch). Default: best')

    # 硬件相关参数
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use (default: auto-select)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device type (default: cuda)')

    # 评估相关参数
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')

    # 攻击方法选择
    parser.add_argument('--pgd10', action='store_true', default=True,
                        help='Evaluate PGD-10 attack')
    parser.add_argument('--pgd20', action='store_true', default=True,
                        help='Evaluate PGD-20 attack')
    parser.add_argument('--pgd50', action='store_true', default=True,
                        help='Evaluate PGD-50 attack')
    parser.add_argument('--cw', action='store_true', default=True,
                        help='Evaluate C&W attack (L-infinity)')
    parser.add_argument('--aa', action='store_true',
                        help='Evaluate AutoAttack (strongest, slow)')
    parser.add_argument('--all-attacks', action='store_true',
                        help='Evaluate all attacks (PGD-10/20/50, C&W, AutoAttack)')

    parser.add_argument('--no-save', action='store_true',
                        help='Do not save evaluation results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed evaluation results')

    args = parser.parse_args()

    print("=" * 70)
    print("Model Evaluation")
    print("=" * 70)

    try:
        exp_folder = find_experiment_folder(args.exp)
        print(f"\n✓ Found experiment folder: {exp_folder}")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return

    try:
        checkpoint_file = find_checkpoint(exp_folder, args.ckpt)
        print(f"✓ Found {args.ckpt} checkpoint: {checkpoint_file.name}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return

    config = load_exp_config(exp_folder)
    config['batch_size'] = args.batch_size
    config['num_workers'] = args.num_workers
    config["data_dir"] = "/home/chenyu/ADV/data"

    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("✗ CUDA not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                device = torch.device(f'cuda:{args.gpu}')
                print(f"✓ Using GPU: {args.gpu}")
            else:
                device = torch.device('cuda')
                print(f"✓ Using default GPU")
    else:
        device = torch.device('cpu')
        print(f"✓ Using CPU")

    print(f"\n✓ Loading {config.get('dataset', 'unknown')} dataset...")
    try:
        _, test_loader = get_loaders(config)
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    print(f"✓ Loading {config.get('model', 'unknown')} model...")
    try:
        model = get_model(config).to(device)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    try:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Loaded checkpoint")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return

    eval_attacks = {
        'natural': True,
        'pgd10': args.pgd10 or args.all_attacks,
        'pgd20': args.pgd20 or args.all_attacks,
        'pgd50': args.pgd50 or args.all_attacks,
        'cw': args.cw or args.all_attacks,
        'aa': args.aa or args.all_attacks,
    }

    print(f"\nEvaluating model...")
    print("-" * 70)

    results = {}

    try:
        results['natural'] = evaluate_natural(model, device, test_loader)
        print(f"✓ Natural Accuracy: {results['natural']:.2f}%")

        if eval_attacks['pgd10']:
            results['pgd10'] = evaluate_pgd_10(model, device, test_loader)
            print(f"✓ PGD-10 Accuracy: {results['pgd10']:.2f}%")

        if eval_attacks['pgd20']:
            results['pgd20'] = evaluate_pgd_20(model, device, test_loader)
            print(f"✓ PGD-20 Accuracy: {results['pgd20']:.2f}%")

        if eval_attacks['pgd50']:
            results['pgd50'] = evaluate_pgd_50(model, device, test_loader)
            print(f"✓ PGD-50 Accuracy: {results['pgd50']:.2f}%")

        if eval_attacks['cw']:
            print("Running C&W attack (L-infinity, eps=8/255)...")
            results['cw'] = evaluate_cw(model, device, test_loader, eps=8/255, alpha=2/255, steps=50)
            print(f"✓ C&W Accuracy: {results['cw']:.2f}%")

        if eval_attacks['aa']:
            print("Running AutoAttack (L-infinity, eps=8/255, this may take a while)...")
            results['aa'] = evaluate_aa(model, device, test_loader, norm='Linf', eps=8/255, verbose=args.verbose)
            print(f"✓ AutoAttack Accuracy: {results['aa']:.2f}%")

    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("Evaluation Results (L-infinity, eps=8/255)")
    print("=" * 70)
    print(f"\n  Natural Accuracy:     {results.get('natural', 0):7.2f}%")

    if 'pgd10' in results:
        print(f"  PGD-10 Accuracy:      {results['pgd10']:7.2f}%")
    if 'pgd20' in results:
        print(f"  PGD-20 Accuracy:      {results['pgd20']:7.2f}%")
    if 'pgd50' in results:
        print(f"  PGD-50 Accuracy:      {results['pgd50']:7.2f}%")
    if 'cw' in results:
        print(f"  C&W Accuracy:         {results['cw']:7.2f}%")
    if 'aa' in results:
        print(f"  AutoAttack Accuracy:  {results['aa']:7.2f}%")

    if 'pgd10' in results:
        robustness_gap = results['natural'] - results['pgd10']
        print(f"\n  Robustness Gap:       {robustness_gap:7.2f}%")

    print("=" * 70)

    if not args.no_save:
        results_file = exp_folder / 'eval_results_{}.txt'.format(args.ckpt)
        with open(results_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Model Evaluation Results\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Experiment: {exp_folder.name}\n")
            f.write(f"Checkpoint Type: {args.ckpt}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {device}\n\n")
            f.write(f"Dataset:              {config.get('dataset', 'unknown')}\n")
            f.write(f"Model:                {config.get('model', 'unknown')}\n")
            f.write(f"Method:               {config.get('method', 'unknown')}\n")
            f.write(f"Perturbation:         {config.get('perturbation', 'unknown')}\n\n")
            f.write("Threat Model: L-infinity (eps=8/255)\n\n")
            f.write("Results:\n")
            f.write(f"  Natural Accuracy:     {results.get('natural', 0):.2f}%\n")

            if 'pgd10' in results:
                f.write(f"  PGD-10 Accuracy:      {results['pgd10']:.2f}%\n")
            if 'pgd20' in results:
                f.write(f"  PGD-20 Accuracy:      {results['pgd20']:.2f}%\n")
            if 'pgd50' in results:
                f.write(f"  PGD-50 Accuracy:      {results['pgd50']:.2f}%\n")
            if 'cw' in results:
                f.write(f"  C&W Accuracy:         {results['cw']:.2f}%\n")
            if 'aa' in results:
                f.write(f"  AutoAttack Accuracy:  {results['aa']:.2f}%\n")

            if 'pgd10' in results:
                robustness_gap = results['natural'] - results['pgd10']
                f.write(f"\n  Robustness Gap:       {robustness_gap:.2f}%\n")

            f.write("\nAttack Details:\n")
            f.write(f"  PGD: epsilon=8/255, alpha=2/255, steps=10/20/50\n")
            f.write(f"  C&W: epsilon=8/255, steps=100, L-infinity\n")
            f.write(f"  AutoAttack: epsilon=8/255, standard version\n")
            f.write("=" * 70 + "\n")

        print(f"\n✓ Results saved to: {results_file}")

    print("=" * 70)


if __name__ == '__main__':
    main()
