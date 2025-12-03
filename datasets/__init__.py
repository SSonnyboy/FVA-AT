from .cifar import get_cifar10_loaders, get_cifar100_loaders
from .svhn import get_svhn_loaders

def get_loaders(config):
    if config.dataset == 'cifar10':
        return get_cifar10_loaders(config)
    elif config.dataset == 'cifar100':
        return get_cifar100_loaders(config)
    elif config.dataset == 'svhn':
        return get_svhn_loaders(config)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
