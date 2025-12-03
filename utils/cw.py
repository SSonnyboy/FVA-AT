import os

import numpy as np
import torch
import torch.nn.functional as F


class CW_Linf:
    def __init__(
        self,
        model,
        eps,
        alpha,
        steps=50,
        restarts=1,
        mu=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        device=None,
    ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.restarts = restarts
        self.device = device if device is not None else next(model.parameters()).device

        # 预计算上下限
        mu = torch.tensor(mu).view(3, 1, 1).to(self.device)
        std = torch.tensor(std).view(3, 1, 1).to(self.device)
        self.std = std
        self.upper_limit = (1 - mu) / std
        self.lower_limit = (0 - mu) / std

    def clamp(self, X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    def CW_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        loss_value = -(
            x[np.arange(x.shape[0]), y]
            - x_sorted[:, -2] * ind
            - x_sorted[:, -1] * (1.0 - ind)
        )
        return loss_value.mean()

    def __call__(self, X, y):
        """
        生成对抗样本，接口类似PGD
        """
        model = self.model
        device = self.device
        epsilon = self.eps / self.std if isinstance(self.eps, float) else self.eps
        alpha = self.alpha / self.std if isinstance(self.alpha, float) else self.alpha

        max_loss = torch.zeros(y.shape[0]).to(device)
        max_delta = torch.zeros_like(X).to(device)

        for zz in range(self.restarts):
            delta = torch.zeros_like(X).to(device)
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(
                    -epsilon[i][0][0].item(), epsilon[i][0][0].item()
                )
            delta.data = self.clamp(delta, self.lower_limit - X, self.upper_limit - X)
            delta.requires_grad = True

            for _ in range(self.steps):
                output = model(X + delta)
                index = torch.where(output.max(1)[1] == y)
                if len(index[0]) == 0:
                    break
                loss = self.CW_loss(output, y)
                loss.backward()
                grad = delta.grad.detach()
                d = delta[index[0], :, :, :]
                g = grad[index[0], :, :, :]
                d = self.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
                d = self.clamp(
                    d,
                    self.lower_limit - X[index[0], :, :, :],
                    self.upper_limit - X[index[0], :, :, :],
                )
                delta.data[index[0], :, :, :] = d
                delta.grad.zero_()

            all_loss = F.cross_entropy(model(X + delta), y, reduction="none").detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

        return X + max_delta  # 返回扰动后的对抗样本
