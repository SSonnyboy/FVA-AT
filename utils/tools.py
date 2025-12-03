import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(config, model):
    return optim.SGD(
        model.parameters(),
        lr=config.lr_init,
        momentum=0.9,
        weight_decay=5e-4
    )

def get_scheduler(config, optimizer):
    return MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1
    )
