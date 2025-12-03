import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from collections import OrderedDict
import random
import numpy as np

EPS = 1e-20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(config, model):
    if config.optimizer == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config.lr_init,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config.lr_init,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

def get_scheduler(config, optimizer):
    if config.lr_scheduler == 'multi_step':
        return MultiStepLR(
            optimizer,
            milestones=config.lr_steps,
            gamma=config.lr_gamma
        )
    elif config.lr_scheduler == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=config.epochs
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
