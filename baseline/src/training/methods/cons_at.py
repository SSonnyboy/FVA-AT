import torch
import torch.nn as nn
from torchvision import transforms
from training.perturbations import get_perturbation
from training.perturbations.utils import (
    compute_all_layer_weights,
    FeatureHookCollector,
    LayerWeightTracker,
)
from attacks import PGD
from losses import consistency_loss


def cons_at_train(
    config,
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    perturbation=None,
    epoch=0,
):
    model.train()
    pgd_attack = PGD(
        eps=config.epsilon / 255,
        alpha=config.alpha / 255,
        steps=config.n_steps,
        loss_fn=criterion,
    )

    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images_aug1, images_aug2 = images[0].to(device), images[1].to(device)
        images_aug = torch.cat([images_aug1, images_aug2], dim=0)
        labels_aug = labels.repeat(2).to(device)

        with torch.no_grad():
            images_adv = pgd_attack(model, images_aug, labels_aug)

        diff = None
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(images_adv, labels_aug)
            perturbation.perturb(diff)

        logits_adv = model(images_adv)
        logits_adv1, logits_adv2 = torch.chunk(logits_adv, 2, dim=0)

        loss_ce = criterion(logits_adv, labels_aug)
        loss_con = consistency_loss(logits_adv1, logits_adv2, config.temperature)
        loss = loss_ce + config.lam * loss_con

        optimizer.zero_grad()
        loss.backward()
        if epoch < 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if diff is not None:
            perturbation.restore(diff)

        total_loss += loss.item()
        _, predicted = logits_adv.max(1)
        correct += predicted.eq(labels_aug).sum().item()
        total += labels_aug.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, accuracy
