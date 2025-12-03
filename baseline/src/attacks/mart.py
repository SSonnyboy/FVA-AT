# The source code is from: https://github.com/YisenWang/MART
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy


def _batch_l2norm(x):
    x_flat = x.view(x.size(0), -1)
    return torch.norm(x_flat, dim=1)


def generate_mart(model, images, labels, eps, alpha, steps):
    model.eval()
    # generate adversarial example
    x_adv = images.detach() + 0.001 * torch.randn(images.shape).cuda().detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = F.cross_entropy(model(x_adv), labels)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, images - eps), images + eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = torch.clamp(x_adv, 0.0, 1.0).clone().detach()
    return x_adv