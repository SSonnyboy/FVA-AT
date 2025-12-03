import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_div(logits1, logits2):
    p = F.log_softmax(logits1, dim=1)
    q = F.softmax(logits2, dim=1)
    return F.kl_div(p, q, reduction='batchmean')


def mart_loss(logits_clean, logits_adv, y, beta=6.0):
    kl = nn.KLDivLoss(reduction='none')
    batch_size = len(logits_clean)
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits_clean, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    return loss_adv, beta * loss_robust
