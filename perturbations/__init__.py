import copy
import torch
from .awp import AWP
from .rwp import RWP


def get_perturbation(config, model, optimizer, device=None):
    """
    Create perturbation object with required models and optimizers.

    Args:
        config: Configuration object
        model: Main model to train
        optimizer: Main optimizer
        device: Device to use (defaults to model's device)

    Returns:
        Perturbation object (AWP/RWP/UAWP) or None
    """
    if config.perturbation == "awp":
        # AWP requires a proxy model and optimizer
        proxy = copy.deepcopy(model)
        # Get learning rate from main optimizer if available
        lr = 0.01  # Default fallback
        proxy_optimizer = torch.optim.SGD(proxy.parameters(), lr=lr)
        return AWP(model, proxy, proxy_optimizer, config)
    elif config.perturbation == "rwp":
        # RWP requires two proxy models and an optimizer for proxy_2
        proxy_1 = copy.deepcopy(model)
        proxy_2 = copy.deepcopy(model)
        # Get learning rate from main optimizer if available
        lr = 0.01  # Default fallback
        proxy_2_optimizer = torch.optim.SGD(proxy_2.parameters(), lr=lr)
        return RWP(model, proxy_1, proxy_2, proxy_2_optimizer, config)
    else:
        return None
