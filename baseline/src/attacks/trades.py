import torch
import torch.nn.functional as F


def kl_div(input, targets, reduction='batchmean'):
    return F.kl_div(F.log_softmax(input, dim=1), F.softmax(targets, dim=1),
                    reduction=reduction)


def generate_trades(model, x, y, eps=8/255, alpha=2/255, steps=10):
    x_adv = x + 0.001 * torch.randn_like(x)
    x_adv = x_adv.detach()
    model.eval()
    logits_clean = model(x).detach()
    for _ in range(steps):
        x_adv.requires_grad = True

        with torch.enable_grad():
            logits_adv = model(x_adv)
            loss = kl_div(logits_adv, logits_clean,reduction='sum')

        grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad)
        x_adv = torch.clamp(x_adv, min=x - eps, max=x + eps)
        x_adv = torch.clamp(x_adv, min=0, max=1)
    model.train()
    return x_adv.detach()
