import torch
import torch.nn as nn
from training.perturbations import get_perturbation
from training.perturbations.utils import (
    compute_all_layer_weights,
    FeatureHookCollector,
    LayerWeightTracker,
)
from attacks import generate_trades
from losses import trades_loss


def trades_train(
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

    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            images_adv = generate_trades(
                model,
                images,
                labels,
                eps=config.epsilon / 255,
                alpha=config.alpha / 255,
                steps=config.n_steps,
            )

        diff = None
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(images_adv, labels)
            perturbation.perturb(diff)

        logits_clean = model(images)
        logits_adv = model(images_adv)
        loss = trades_loss(logits_clean, logits_adv, labels, config.beta)

        optimizer.zero_grad()
        loss.backward()
        if epoch < 1:
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
