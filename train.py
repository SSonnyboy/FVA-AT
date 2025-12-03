import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from attacks import *
from utils.tools import *
from losses import *
from datasets.idx import *
from datasets.tinynet import *
from models import get_model
from utils.extractor import *
from utils.rescore import rescore
from utils.eps_controller import EPS_ruler
from torchattacks import PGD as PGD_eval
from perturbations import get_perturbation
# ============ 权重分布可视化函数 ============
def visualize_score_distribution(score, epoch, save_dir):
    """
    可视化样本权重分布

    Args:
        score: 权重分数 [batch_size]
        epoch: 当前epoch
        save_dir: 保存目录
    """
    score_np = score.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 直方图
    axes[0].hist(score_np, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Weight Score', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f'Weight Distribution - Epoch {epoch}', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 统计信息
    axes[1].axis('off')
    stats_text = f"""
    📊 Weight Statistics (Epoch {epoch})

    Mean:        {score_np.mean():.6f}
    Std:         {score_np.std():.6f}
    Min:         {score_np.min():.6f}
    Max:         {score_np.max():.6f}
    Median:      {np.median(score_np):.6f}

    Q1 (25%):    {np.percentile(score_np, 25):.6f}
    Q3 (75%):    {np.percentile(score_np, 75):.6f}

    Batch Size:  {len(score_np)}
    """
    axes[1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # 保存文件
    weight_dir = Path(save_dir) / 'weight_distributions'
    weight_dir.mkdir(parents=True, exist_ok=True)
    save_path = weight_dir / f'weight_dist_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


# ============ 断点续训函数 ============
def save_checkpoint(checkpoint_path, epoch, model, optimizer, feature_layer, opt_feature_layer,
                   scheduler, best_pgd_acc, best_pgd_epoch, csv_path=None):
    """
    保存训练检查点

    Args:
        checkpoint_path: 检查点保存路径
        epoch: 当前epoch
        model: 主模型
        optimizer: 主优化器
        feature_layer: 特征提取层
        opt_feature_layer: 特征层优化器
        scheduler: 学习率调度器
        best_pgd_acc: 最佳对抗准确率
        best_pgd_epoch: 最佳对抗准确率的epoch
        csv_path: 结果CSV路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'feature_layer_state_dict': feature_layer.state_dict(),
        'opt_feature_layer_state_dict': opt_feature_layer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_pgd_acc': best_pgd_acc,
        'best_pgd_epoch': best_pgd_epoch,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved to {checkpoint_path}")

    return checkpoint


def load_checkpoint(checkpoint_path, model, optimizer, feature_layer, opt_feature_layer, scheduler, device):
    """
    加载训练检查点

    Args:
        checkpoint_path: 检查点路径
        model: 主模型
        optimizer: 主优化器
        feature_layer: 特征提取层
        opt_feature_layer: 特征层优化器
        scheduler: 学习率调度器
        device: 设备

    Returns:
        start_epoch, best_pgd_acc, best_pgd_epoch
    """
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 0, 0.0, 0

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    feature_layer.load_state_dict(checkpoint['feature_layer_state_dict'])
    opt_feature_layer.load_state_dict(checkpoint['opt_feature_layer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_pgd_acc = checkpoint['best_pgd_acc']
    best_pgd_epoch = checkpoint['best_pgd_epoch']

    print(f"✓ Checkpoint loaded from {checkpoint_path}")
    print(f"  Resuming from epoch {start_epoch}")
    print(f"  Best PGD-10 accuracy so far: {best_pgd_acc:.2f}% (epoch {best_pgd_epoch})")

    return start_epoch, best_pgd_acc, best_pgd_epoch


parser = argparse.ArgumentParser(description='Adversarial Training')

parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'svhn'],
                    help='Dataset name')
parser.add_argument('--model', type=str, default='resnet18',
                    choices=['resnet18', 'wrn34_10'],
                    help='Model architecture')

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr_init', type=float, default=0.1)

parser.add_argument('--epsilon', type=float, default=8.)
parser.add_argument('--alpha', type=float, default=2.)
parser.add_argument('--n_steps', type=int, default=10)
parser.add_argument('--beta', type=float, default=6.)

parser.add_argument('--perturbation', type=str, default="awp",
                    choices=['none', 'awp', 'rwp'],
                    help='Weight perturbation method')
parser.add_argument('--gamma', type=float, default=0.01)
parser.add_argument('--warmup', type=int, default=0)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--pin_memory', action='store_true')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='GPU ID to use (default: auto-select)')

parser.add_argument('--mode', type=str, default="at", choices=["at", "trades", "mart"])
parser.add_argument('--T', type=float, default=5)
parser.add_argument('--rew_epoch', type=int, default=101)
parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--max_eps', type=float, default=16.,
                    help='Maximum epsilon for dynamic eps scaling')

parser.add_argument('--data_dir', type=str, default='/home/chenyu/ADV/data')
parser.add_argument('--save_dir', type=str, default='./outputs')
parser.add_argument('--exp', type=str, default='test_wow')

# 断点续训参数
parser.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint for resuming training')
parser.add_argument('--save_imd', action='store_true',
                    help='Save intermediate checkpoint at epoch 90')

config = parser.parse_args()

set_seed(config.seed)

device = torch.device(f"cuda:{config.gpu_id}")
torch.cuda.set_device(config.gpu_id)

if config.dataset == "cifar10":
    train_loader, test_loader, set_lib = get_cifar10_loaders_idx(config)
    config.num_classes = 10
elif config.dataset == "cifar100":
    train_loader, test_loader, set_lib = get_cifar100_loaders_idx(config)
    config.num_classes = 100
elif config.dataset == "svhn":
    train_loader, test_loader, set_lib = get_svhn_loaders_idx(config)
    config.num_classes = 10
elif config.dataset == "tinynet":
    train_loader, test_loader, set_lib = get_tinynet_loaders_idx(config)
    config.num_classes = 200

config.exp_name = "{}_{}_{}_{}".format(config.dataset, config.model, config.perturbation, config.mode)
config.save_dir = os.path.join(config.save_dir, config.exp_name)
Path(config.save_dir).mkdir(parents=True, exist_ok=True)

config_dict = vars(config)
config_json_path = os.path.join(config.save_dir, 'config.json')
with open(config_json_path, 'w') as f:
    json.dump(config_dict, f, indent=4)

model = get_model(config).to(device)
optimizer = get_optimizer(config, model)
scheduler = get_scheduler(config, optimizer)
criterion = nn.CrossEntropyLoss()

feature_layer = FeatureLayer(config=config).to(device)
opt_feature_layer = torch.optim.Adam(feature_layer.parameters(), lr=0.001)

best_pgd_acc = 0.0
best_pgd_epoch = 0
start_epoch = 0

# ============ 断点续训逻辑 ============
if config.resume is not None:
    start_epoch, best_pgd_acc, best_pgd_epoch = load_checkpoint(
        config.resume, model, optimizer, feature_layer, opt_feature_layer, scheduler, device
    )
    print(f"🔄 断点续训模式已启用")
    print(f"   将从 epoch {start_epoch} 继续训练")

# CSV处理
csv_path = os.path.join(config.save_dir, 'results.csv')
if config.resume is not None and Path(csv_path).exists():
    # 断点续训：追加到现有CSV
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    print(f"📝 追加到现有CSV: {csv_path}")
else:
    # 新训练：创建新CSV
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Avg_Loss', 'Avg_Main_Loss', 'Avg_Entropy_Loss', 'Clean_Acc', 'PGD10_Acc'])
    csv_file.flush()
    print(f"📊 创建新CSV: {csv_path}")

attacker = PGD_eval(
    model,
    eps=config.epsilon / 255,
    alpha=config.alpha / 255,
    steps=config.n_steps,
)

if config.perturbation and config.perturbation != "none":
    perturbation = get_perturbation(config, model, optimizer, device)
else:
    perturbation = None

# init eps controller
eps_ruler = EPS_ruler(set_lib, config.epsilon / 255, config.max_eps / 255, config.epochs, config.rew_epoch, device, config.T)

print(f"\n{'='*70}")
print(f"开始训练 | Epochs: {start_epoch} → {config.epochs-1}")
print(f"{'='*70}\n")
for epoch in range(start_epoch, config.epochs):
    model.train()
    pgd_attack = PGD(
        eps=config.epsilon / 255,
        alpha=config.alpha / 255,
        steps=config.n_steps,
    )
    epoch_loss = 0.0
    epoch_main_loss = 0.0
    epoch_entropy_loss = 0.0
    eps_ruler.update_sur_max_eps(epoch)

    is_first_batch = True  # 标记第一个batch

    for i, (images, labels, index) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Use per-sample eps only after rew_epoch
        if epoch > config.rew_epoch:
            eps = eps_ruler.get_eps(index)
        else:
            eps = None

        with torch.no_grad():
            if config.mode == "trades":
                images_adv = generate_trades(
                    model,
                    images,
                    labels,
                    eps=config.epsilon / 255,
                    alpha=config.alpha / 255,
                    steps=config.n_steps,eps_per_sample=eps
                )
            else:
                images_adv = pgd_attack(model, images, labels, eps_per_sample=eps)
        diff = None
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(images_adv, labels)
            perturbation.perturb(diff)

        logits_clean, feats_clean = model(images, feature=True)
        logits_adv, feats_adv = model(images_adv, feature=True)

        # update extractor
        clean_embeding, out_clean = feature_layer(feats_clean.detach())
        adv_embeding, out_adv = feature_layer(feats_adv.detach())
        opt_feature_layer.zero_grad()
        loss_ex = criterion(out_clean, labels)
        loss_ex.backward()
        opt_feature_layer.step()

        # reweight loss
        score, s = rescore(out_clean.detach(),out_adv.detach(), labels, clean_embeding.detach(), adv_embeding.detach(), cnt=epoch-config.rew_epoch, T=config.T)
        # score, s = rescore(logits_clean.detach(),logits_adv.detach(), labels, logits_clean.detach(), logits_adv.detach(), cnt=epoch-config.rew_epoch, T=config.T)

        # ============ 权重分布可视化 (第一个batch，每10个epoch保存一张) ============
        if is_first_batch and epoch % 10 == 0:
            visualize_score_distribution(score, epoch, config.save_dir)

        is_first_batch = False  # 更新标记

        eps_ruler.update_lib(index, s)
        
        if config.mode == "mart":
            is_rew = epoch > config.rew_epoch
            loss_main = mart_loss(logits_clean, logits_adv, labels, config.beta, is_rew, score)
        elif config.mode == "trades":
            is_rew = epoch > config.rew_epoch
            loss_main = trades_loss(logits_clean, logits_adv, labels, config.beta, is_rew, score)
        else:
            if epoch > config.rew_epoch:
                loss_main = F.cross_entropy(logits_adv, labels, reduction = 'none')
                loss_main = (loss_main * score).mean()
            else:
                loss_main = criterion(logits_adv, labels)
            

        probs_adv = torch.softmax(logits_adv, dim=1)
        entropy = -(probs_adv * torch.log(probs_adv + 1e-8)).sum(dim=1)
        loss_entropy = entropy.mean()
        loss = loss_main

        optimizer.zero_grad()
        loss.backward()
        if epoch < 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_main_loss += loss_main.item()
        epoch_entropy_loss += loss_entropy.item()

        if diff is not None:
            perturbation.restore(diff)

    scheduler.step()

    avg_loss = epoch_loss / len(train_loader)
    avg_main_loss = epoch_main_loss / len(train_loader)
    avg_entropy_loss = epoch_entropy_loss / len(train_loader)

    model.eval()
    clean_correct = 0
    pgd_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, index in test_loader:
            images, labels = images.to(device), labels.to(device)

            logits_clean = model(images)
            clean_pred = logits_clean.argmax(dim=1)
            clean_correct += (clean_pred == labels).sum().item()
            with torch.enable_grad():
                img_adv = attacker(images, labels)
            logits_adv = model(img_adv)
            pgd_pred = logits_adv.argmax(dim=1)
            pgd_correct += (pgd_pred == labels).sum().item()

            total += labels.size(0)

    clean_acc = 100.0 * clean_correct / total
    pgd_acc = 100.0 * pgd_correct / total

    csv_writer.writerow([epoch, f'{avg_loss:.4f}', f'{avg_main_loss:.4f}', f'{avg_entropy_loss:.4f}', f'{clean_acc:.2f}', f'{pgd_acc:.2f}'])
    csv_file.flush()

    print(f"Epoch {epoch}: Loss={avg_loss:.4f} (Main={avg_main_loss:.4f}, Entropy={avg_entropy_loss:.4f}), Clean_Acc={clean_acc:.2f}%, PGD10_Acc={pgd_acc:.2f}%")

    if pgd_acc > best_pgd_acc:
        best_pgd_acc = pgd_acc
        best_pgd_epoch = epoch
        torch.save(model.state_dict(), os.path.join(config.save_dir, 'best_pgd_model.pth'))

    torch.save(model.state_dict(), os.path.join(config.save_dir, 'latest_model.pth'))

    # ============ 保存中间检查点 (Epoch 99) ============
    if config.save_imd and epoch == 99:
        checkpoint_path = os.path.join(config.save_dir, 'checkpoint_epoch99.pt')
        save_checkpoint(
            checkpoint_path, epoch, model, optimizer, feature_layer,
            opt_feature_layer, scheduler, best_pgd_acc, best_pgd_epoch
        )
        print(f"✓ 中间检查点已保存 (epoch 99)")

csv_file.close()
print(f"\nBest PGD10 accuracy: {best_pgd_acc:.2f}% at epoch {best_pgd_epoch}")
print(f"Results saved to {csv_path}")