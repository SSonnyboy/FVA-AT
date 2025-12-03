import torch
import torch.nn.functional as F


def js_divergence(logits1, logits2, temperature=1.0):
    p = F.softmax(logits1 / temperature, dim=1)
    q = F.softmax(logits2 / temperature, dim=1)

    m = 0.5 * (p + q)
    kl_pm = F.kl_div(torch.log(m.clamp(min=1e-8)), p, reduction='batchmean')
    kl_qm = F.kl_div(torch.log(m.clamp(min=1e-8)), q, reduction='batchmean')

    return 0.5 * kl_pm + 0.5 * kl_qm


def consistency_loss(logits_adv1, logits_adv2, temperature=1.0):
    return js_divergence(logits_adv1, logits_adv2, temperature)
