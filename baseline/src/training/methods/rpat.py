import torch
import torch.nn as nn
from training.perturbations import get_perturbation
from training.perturbations.utils import (
    compute_all_layer_weights,
    FeatureHookCollector,
    LayerWeightTracker,
)
from attacks import PGD

criterion_ra = nn.MSELoss()


def rpat_train(
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
        # images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        benign_images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            images_adv = pgd_attack(model, benign_images, labels)

        # Apply weight perturbation if available and after warmup period
        diff = None
        if perturbation is not None and epoch >= perturbation.warmup:
            diff = perturbation.calc_awp(images_adv, labels)
            perturbation.perturb(diff)
        adv_outputs = model(images_adv)
        loss_adv_ce = criterion(adv_outputs, labels)
        loss = loss_adv_ce

        ### Robust Perception loss ###
        if epoch >= config.RA_start:
            interpolation_rate = config.RA_ip_rate
            interpolation_images = (
                interpolation_rate * benign_images
                + (1 - interpolation_rate) * images_adv
            )
            interpolation_output_1 = model(interpolation_images)

            benign_outputs = model(benign_images)
            interpolation_output_2 = (
                interpolation_rate * benign_outputs
                + (1 - interpolation_rate) * adv_outputs
            )

            loss_ra = criterion_ra(interpolation_output_1, interpolation_output_2)
            loss += config.lam * loss_ra

        optimizer.zero_grad()
        loss.backward()
        if epoch < 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # Restore original weights
        if diff is not None:
            perturbation.restore(diff)

        total_loss += loss.item()
        _, predicted = adv_outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, accuracy
