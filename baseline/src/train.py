import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import time

# Ensure src is in path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from common.config import Config
from common.args import get_args
from common.utils import set_seed, get_optimizer, get_scheduler
from datasets import get_loaders
from models import get_model
from training.methods import get_train_fn
from training.perturbations import get_perturbation
from losses import get_criterion
from evals import evaluate_natural, evaluate_pgd_10, evaluate_pgd_20
from evals.pgd import evaluate_pgd_10_classwise
from utils import Logger, save_checkpoint


def main():
    args = get_args()
    config = Config()
    config.load_from_args(args)
    print("Configuration:")
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")
    set_seed(config.seed)

    if config.device == "cuda" and torch.cuda.is_available():
        if config.gpu_id is not None:
            device = torch.device(f"cuda:{config.gpu_id}")
            torch.cuda.set_device(config.gpu_id)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 生成实验名称和文件夹结构
    # 主文件夹格式: {dataset}_{model}_{method}_{perturbation}
    # 子文件夹格式: seed_{seed}_{timestamp}
    main_exp_dir = (
        f"{config.dataset}_{config.model}_{config.method}_{config.perturbation}"
    )
    if config.exp_name is not None:
        sub_exp_dir = f"seed_{config.seed}_{config.exp_name}"
    else:
        sub_exp_dir = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if config.exp_name is None:
        config.exp_name = sub_exp_dir

    # 确保 out_dir 相对于项目根目录（FAWP 文件夹），而不是脚本所在的 src 文件夹
    # 如果 out_dir 是相对路径，将其转换为相对于项目根目录的绝对路径
    out_dir = Path(config.out_dir)
    if not out_dir.is_absolute():
        # 获取项目根目录（src 的父目录）
        project_root = Path(__file__).parent.parent
        out_dir = project_root / out_dir

    # 创建 Logger 时传入主文件夹和子文件夹
    logger = Logger(str(out_dir), main_exp_dir, sub_exp_dir)
    checkpoint_dir = logger.get_checkpoint_dir()

    train_loader, test_loader = get_loaders(config)
    model = get_model(config).to(device)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    criterion = get_criterion(config)

    if config.perturbation and config.perturbation != "none":
        perturbation = get_perturbation(config, model, optimizer, device)
    else:
        perturbation = None

    train_fn = get_train_fn(config)

    start_epoch = 0
    best_pgd10_acc = 0

    if config.resume:
        from utils import load_checkpoint

        start_epoch, best_pgd10_acc = load_checkpoint(model, optimizer, config.resume)

    print(f"Training {config.method} on {config.dataset} with {config.model}")
    print(f"Perturbation: {config.perturbation}")
    print(f"Device: {device}")
    print(f"Experiment: {config.exp_name}")

    # 记录训练开始时间
    training_start_time = time.time()

    for epoch in range(start_epoch, config.epochs):

        train_loss, train_acc = train_fn(
            config,
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            perturbation,
            epoch,
        )

        scheduler.step()

        test_acc = evaluate_natural(model, device, test_loader)
        pgd10_acc = evaluate_pgd_10(model, device, test_loader)

        logger.log_metrics(epoch, train_loss, train_acc, test_acc, pgd10_acc)

        if pgd10_acc > best_pgd10_acc:
            best_pgd10_acc = pgd10_acc
            save_checkpoint(model, optimizer, epoch, best_pgd10_acc, checkpoint_dir)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Test Acc: {test_acc:.2f}% | PGD-10 Acc: {pgd10_acc:.2f}%"
            )

    save_checkpoint(
        model, optimizer, epoch, test_acc, checkpoint_dir, filename="final_model.pt"
    )

    # 计算总训练时间
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    print(f"\nTraining completed.")
    print(f"  Best PGD-10 Accuracy:  {best_pgd10_acc:.2f}%")
    print(
        f"Total training time: {total_training_time:.2f}s ({total_training_time/3600:.2f}h)"
    )

    # 训练完成后生成最终报告和图表（传入总训练时间）
    logger.finalize(total_training_time=total_training_time)


if __name__ == "__main__":
    main()
