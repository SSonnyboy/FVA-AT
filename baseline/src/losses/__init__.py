import torch.nn as nn
from .trades import trades_loss, kl_div
from .mart import mart_loss
from .consistency import consistency_loss


def get_criterion(config):
    if config.method == 'pgd_at':
        return nn.CrossEntropyLoss()
    elif config.method == 'trades':
        return nn.CrossEntropyLoss()
    elif config.method == 'mart':
        return nn.CrossEntropyLoss()
    elif config.method == 'cons_at':
        return nn.CrossEntropyLoss()
    else:
        return nn.CrossEntropyLoss()
