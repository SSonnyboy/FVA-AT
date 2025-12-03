import torch
import torch.nn as nn


class PGD:
    def __init__(self, eps=8/255, alpha=2/255, steps=10, random_start=True, loss_fn=None):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def __call__(self, model, x, y):
        x_adv = x.clone().detach()

        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(
                -self.eps, self.eps
            )
            x_adv = torch.clamp(x_adv, min=0, max=1).detach()

        for _ in range(self.steps):
            x_adv.requires_grad = True

            with torch.enable_grad():
                logits = model(x_adv)
                loss = self.loss_fn(logits, y)

            grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
            x_adv = x_adv.detach() + self.alpha * torch.sign(grad)
            x_adv = torch.clamp(x_adv, min=x - self.eps, max=x + self.eps)
            x_adv = torch.clamp(x_adv, min=0, max=1)

        return x_adv.detach()
