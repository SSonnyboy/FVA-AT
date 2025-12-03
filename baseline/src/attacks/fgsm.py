import torch
import torch.nn as nn


class FGSM:
    def __init__(self, eps=8/255, loss_fn=None):
        self.eps = eps
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def __call__(self, model, x, y):
        x_adv = x.clone().detach()
        x_adv.requires_grad = True

        with torch.enable_grad():
            logits = model(x_adv)
            loss = self.loss_fn(logits, y)

        grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
        x_adv = x_adv.detach() + self.eps * torch.sign(grad)
        x_adv = torch.clamp(x_adv, min=x - self.eps, max=x + self.eps)
        x_adv = torch.clamp(x_adv, min=0, max=1)

        return x_adv
