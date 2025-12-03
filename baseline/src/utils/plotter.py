import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


class MetricsPlotter:
    """实时绘制训练指标曲线"""

    def __init__(self, save_dir):
        """
        Args:
            save_dir: 保存图表的目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化存储数据
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'pgd10_acc': []
        }

    def update(self, epoch, train_loss, train_acc, test_acc, pgd10_acc):
        """更新指标数据并绘制曲线

        Args:
            epoch: 当前epoch
            train_loss: 训练损失
            train_acc: 训练精度
            test_acc: 测试精度
            pgd10_acc: PGD-10对抗精度
        """
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['test_acc'].append(test_acc)
        self.metrics['pgd10_acc'].append(pgd10_acc)

        # 每个epoch后绘制曲线
        self._plot_metrics()

    def _plot_metrics(self):
        """绘制所有指标曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')

        epochs = self.metrics['epoch']

        # 1. 训练损失曲线
        ax = axes[0, 0]
        ax.plot(epochs, self.metrics['train_loss'], 'b-o', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2. 训练精度曲线
        ax = axes[0, 1]
        ax.plot(epochs, self.metrics['train_acc'], 'g-o', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title('Training Accuracy', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        # 3. 测试精度 vs PGD-10精度对比
        ax = axes[1, 0]
        ax.plot(epochs, self.metrics['test_acc'], 'r-s', linewidth=2, markersize=4, label='Test Acc')
        ax.plot(epochs, self.metrics['pgd10_acc'], 'orange', marker='^', linewidth=2, markersize=4, label='PGD-10 Acc')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title('Natural vs Adversarial Accuracy', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

        # 4. 鲁棒性差距
        ax = axes[1, 1]
        robustness_gap = [t - p for t, p in zip(self.metrics['test_acc'], self.metrics['pgd10_acc'])]
        ax.plot(epochs, robustness_gap, 'purple', marker='d', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Gap (%)', fontsize=11)
        ax.set_title('Robustness Gap (Test Acc - PGD-10 Acc)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(robustness_gap) * 1.1 if robustness_gap else 100])

        plt.tight_layout()

        # 保存图表
        plot_path = self.save_dir / 'metrics.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"📊 Metrics plot saved to {plot_path}")

    def plot_final_summary(self):
        """绘制最终总结图表（可选的额外可视化）"""
        if len(self.metrics['epoch']) == 0:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        epochs = self.metrics['epoch']
        x = np.arange(len(epochs))
        width = 0.25

        # 最后一个epoch的各项指标
        metrics_names = ['Train Acc', 'Test Acc', 'PGD-10 Acc']
        final_metrics = [
            self.metrics['train_acc'][-1],
            self.metrics['test_acc'][-1],
            self.metrics['pgd10_acc'][-1]
        ]

        bars = ax.bar(metrics_names, final_metrics, color=['green', 'blue', 'orange'], alpha=0.8, width=0.6)

        # 在柱子上显示数值
        for bar, val in zip(bars, final_metrics):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Final Metrics (Epoch {epochs[-1]})', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        summary_path = self.save_dir / 'summary.png'
        plt.savefig(summary_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"📊 Summary plot saved to {summary_path}")

    def get_best_metrics(self):
        """获取最佳指标"""
        if len(self.metrics['epoch']) == 0:
            return {}

        best_test_acc_idx = np.argmax(self.metrics['test_acc'])
        best_pgd10_acc_idx = np.argmax(self.metrics['pgd10_acc'])

        return {
            'best_test_acc': self.metrics['test_acc'][best_test_acc_idx],
            'best_test_acc_epoch': self.metrics['epoch'][best_test_acc_idx],
            'best_pgd10_acc': self.metrics['pgd10_acc'][best_pgd10_acc_idx],
            'best_pgd10_acc_epoch': self.metrics['epoch'][best_pgd10_acc_idx],
            'final_train_loss': self.metrics['train_loss'][-1],
            'final_train_acc': self.metrics['train_acc'][-1],
        }
