import os
import csv
from pathlib import Path
from .plotter import MetricsPlotter


class Logger:
    def __init__(self, log_dir, main_exp_dir, sub_exp_dir=None):
        """
        初始化Logger，支持两级文件夹结构

        Args:
            log_dir: 基础输出目录
            main_exp_dir: 主文件夹名 (e.g., "{dataset}_{model}_{method}_{perturbation}")
            sub_exp_dir: 子文件夹名 (e.g., "seed_{seed}_{timestamp}"), 如果为None则只使用main_exp_dir
        """
        if sub_exp_dir:
            self.log_dir = Path(log_dir) / main_exp_dir / sub_exp_dir
        else:
            self.log_dir = Path(log_dir) / main_exp_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.log_dir / 'metrics.csv'
        self.results_file = self.log_dir / 'eval_results.txt'

        self.metrics_data = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'pgd10_acc': [],
            'robust_loss': []
        }

        self.fieldnames = list(self.metrics_data.keys())
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

        # 初始化plotter用于实时绘制
        self.plotter = MetricsPlotter(self.log_dir)

    def log_metrics(self, epoch, train_loss, train_acc, test_acc, pgd10_acc, robust_loss=None):
        self.metrics_data['epoch'].append(epoch)
        self.metrics_data['train_loss'].append(train_loss)
        self.metrics_data['train_acc'].append(train_acc)
        self.metrics_data['test_acc'].append(test_acc)
        self.metrics_data['pgd10_acc'].append(pgd10_acc)
        self.metrics_data['robust_loss'].append(robust_loss or 0)

        row = {
            'epoch': epoch,
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.2f}',
            'test_acc': f'{test_acc:.2f}',
            'pgd10_acc': f'{pgd10_acc:.2f}',
            'robust_loss': f'{robust_loss:.4f}' if robust_loss else 'N/A'
        }

        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

        # 更新plotter并绘制曲线
        self.plotter.update(epoch, train_loss, train_acc, test_acc, pgd10_acc)

    def log_eval_results(self, results_dict):
        with open(self.results_file, 'w') as f:
            for key, value in results_dict.items():
                f.write(f'{key}: {value}\n')

    def get_checkpoint_dir(self):
        ckpt_dir = self.log_dir / 'checkpoints'
        ckpt_dir.mkdir(exist_ok=True)
        return ckpt_dir

    def finalize(self, total_training_time=None):
        """
        训练完成时调用，生成最终报告和图表

        Args:
            total_training_time: 总训练时间（秒）
        """
        # 绘制最终总结
        self.plotter.plot_final_summary()

        # 生成最终报告
        best_metrics = self.plotter.get_best_metrics()
        report_path = self.log_dir / 'training_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("Final Training Results:\n")
            final_loss = best_metrics.get('final_train_loss', 'N/A')
            final_loss_str = f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else str(final_loss)
            f.write(f"  Final Training Loss:     {final_loss_str}\n")

            final_acc = best_metrics.get('final_train_acc', 'N/A')
            final_acc_str = f"{final_acc:.2f}%" if isinstance(final_acc, (int, float)) else str(final_acc)
            f.write(f"  Final Training Accuracy: {final_acc_str}\n\n")

            f.write("Best Results:\n")
            best_test_acc = best_metrics.get('best_test_acc', 'N/A')
            best_test_acc_epoch = best_metrics.get('best_test_acc_epoch', 'N/A')
            best_test_acc_str = f"{best_test_acc:.2f}%" if isinstance(best_test_acc, (int, float)) else str(best_test_acc)
            f.write(f"  Best Test Accuracy:      {best_test_acc_str} (Epoch {best_test_acc_epoch})\n")

            best_pgd10_acc = best_metrics.get('best_pgd10_acc', 'N/A')
            best_pgd10_acc_epoch = best_metrics.get('best_pgd10_acc_epoch', 'N/A')
            best_pgd10_acc_str = f"{best_pgd10_acc:.2f}%" if isinstance(best_pgd10_acc, (int, float)) else str(best_pgd10_acc)
            f.write(f"  Best PGD-10 Accuracy:    {best_pgd10_acc_str} (Epoch {best_pgd10_acc_epoch})\n")

            # 计算鲁棒性间隙
            if isinstance(best_test_acc, (int, float)) and isinstance(best_pgd10_acc, (int, float)):
                robustness_gap = best_test_acc - best_pgd10_acc
                f.write(f"  Robustness Gap:          {robustness_gap:.2f}%\n\n")
            else:
                f.write(f"  Robustness Gap:          N/A\n\n")

            # 添加总训练时间
            if total_training_time is not None:
                hours = int(total_training_time // 3600)
                minutes = int((total_training_time % 3600) // 60)
                seconds = int(total_training_time % 60)
                f.write("Training Time:\n")
                f.write(f"  Total Time: {total_training_time:.2f}s ({hours}h {minutes}m {seconds}s)\n\n")

            f.write("Output Files:\n")
            f.write(f"  - Metrics CSV:  {self.metrics_file.name}\n")
            f.write(f"  - Metrics Plot: metrics.png\n")
            f.write(f"  - Summary Plot: summary.png\n")
            f.write(f"  - Checkpoints:  checkpoints/\n")
            f.write("=" * 60 + "\n")

        print(f"\n📄 Training report saved to {report_path}")
