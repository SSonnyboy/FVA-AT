import argparse


def get_args():
    parser = argparse.ArgumentParser(description='UAWP Adversarial Training')

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn'],
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='preactresnet18',
                        choices=['resnet18', 'wrn28_10', 'wrn34_10', 'preactresnet18'],
                        help='Model architecture')
    parser.add_argument('--config', type=str, default='pgd_at',
                        help='Training config name (in configs/training/)')

    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr_init', type=float, default=None)
    parser.add_argument('--lr_scheduler', type=str, default=None,
                        choices=['multi_step', 'cosine', 'cyclic'])
    parser.add_argument('--optimizer', type=str, default=None,
                        choices=['sgd', 'adam'])
    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)

    parser.add_argument('--epsilon', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--n_steps', type=int, default=None)
    parser.add_argument('--beta', type=float, default=None)

    parser.add_argument('--perturbation', type=str, default=None,
                        choices=['none', 'awp', 'rwp', 'cwp', 'uawp'],
                        help='Weight perturbation method')

    parser.add_argument('--awp_gamma', type=float, default=None)
    parser.add_argument('--awp_warmup', type=int, default=None)
    parser.add_argument('--rwp_gamma', type=float, default=None)
    parser.add_argument('--rwp_warmup', type=int, default=None)
    parser.add_argument('--cwp_gamma', type=float, default=None)
    parser.add_argument('--cwp_warmup', type=int, default=None)
    parser.add_argument('--cwp_top_k_ratio', type=float, default=None)
    parser.add_argument('--cwp_mode', type=str, default=None,
                        choices=['easy', 'hard'])

    parser.add_argument('--uawp_gamma', type=float, default=None,
                        help='UAWP weight perturbation coefficient')
    parser.add_argument('--uawp_warmup', type=int, default=None,
                        help='UAWP warmup epochs')
    parser.add_argument('--uawp_lr', type=float, default=None,
                        help='UAWP proxy model learning rate')
    parser.add_argument('--uawp_kl_sensitivity', type=float, default=None,
                        help='UAWP KL divergence sensitivity (default: 1.0)')
    parser.add_argument('--uawp_iterations', type=int, default=None,
                        help='UAWP weight perturbation iterations (default: 10)')

    parser.add_argument('--lam', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='GPU ID to use (default: auto-select)')
    parser.add_argument('--out_dir', type=str, default='outputs/',
                        help='Output directory')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume from')

    return parser.parse_args()
