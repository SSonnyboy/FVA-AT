import torch
import torch.nn as nn
from training.perturbations import get_perturbation
from training.perturbations.utils import (
    compute_all_layer_weights,
    FeatureHookCollector,
    LayerWeightTracker,
)
from losses import mart_loss
from attacks import PGD


def mart_train(
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
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            images_adv = pgd_attack(model, images, labels)

        diff = None
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(images_adv, labels)
            perturbation.perturb(diff)

        logits_clean = model(images)
        logits_adv = model(images_adv)
        loss_adv, loss_robust = mart_loss(logits_clean, logits_adv, labels, config.beta)
        loss = loss_adv + loss_robust

        optimizer.zero_grad()
        loss.backward()
        if epoch < 1:
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if diff is not None:
            perturbation.restore(diff)

        total_loss += loss.item()
        _, predicted = logits_clean.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, accuracy
