import torch
import torch.nn as nn
from training.perturbations import get_perturbation
from training.perturbations.utils import (
    compute_all_layer_weights,
    FeatureHookCollector,
    LayerWeightTracker,
)
from attacks import PGD
from collections import OrderedDict
import random

# _layer_weight_tracker = LayerWeightTracker()


def pgd_at_train(
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
        diff = None
        with torch.no_grad():
            images = pgd_attack(model, images, labels)
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(images, labels)
            perturbation.perturb(diff)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if epoch < 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if diff is not None:
            perturbation.restore(diff)

        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, accuracy


# def get_layer_weight_tracker():
#     return _layer_weight_tracker


# def plot_layer_weights_heatmap(save_path=None, title="PGD-AT-Layer-Weights-Heatmap"):
#     _layer_weight_tracker.plot_heatmap(save_path=save_path, title=title)
