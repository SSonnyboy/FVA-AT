import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_div(logits1, logits2):
    p = F.log_softmax(logits1, dim=1)
    q = F.softmax(logits2, dim=1)
    return F.kl_div(p, q, reduction='batchmean')


def trades_loss(logits_clean, logits_adv, y, beta=6.0):
    loss_ce = nn.CrossEntropyLoss()(logits_clean, y)
    loss_kl = kl_div(logits_adv, logits_clean)
    return loss_ce + beta * loss_kl
