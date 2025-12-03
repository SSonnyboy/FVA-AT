import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def rescore(logits, logits_adv, target, clean_embeding, adv_embeding, cnt, T=1):
    N = clean_embeding.shape[0]
    kl = nn.KLDivLoss(reduction='none')
    adv_kl = F.softmax(clean_embeding, dim=1)
    nat_kl = F.softmax(adv_embeding, dim=1)
    nat_probs = F.softmax(logits, dim=1)
    divg = torch.sum(kl(torch.log(adv_kl + 1e-12), nat_kl), dim =1) 
    true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()
    reweight =  float(T) * torch.exp(-T * true_probs)* divg 
    if cnt > 10:
        reweight_unnorm = torch.exp((reweight - reweight.max()))  # numerical stability
        normalized_reweight = reweight_unnorm * N / reweight_unnorm.sum()
        reweight = normalized_reweight.detach()

    N, C = logits_adv.shape
    adv_probs = F.softmax(logits_adv, dim=1)  # [N, C]
    true_probs = adv_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [N]
    mask = torch.arange(C, device=target.device).unsqueeze(0) != target.unsqueeze(1)  # [N, C]
    non_target_probs = adv_probs * mask  
    topk_vals, _ = torch.topk(non_target_probs, k=3, dim=1)  # [N, cal]
    margin = (true_probs.unsqueeze(1) - topk_vals).sum(dim=1)  # [N]

    return reweight, margin

