"""
Layer Weight Calculation Utilities (Innovation Point 2: Uncertainty-Driven Layer Weights)

Simplified version with direct KL divergence to weight mapping.

This module contains layer weight calculation functions for applying adaptive
perturbation strengths to different layers based on their feature uncertainty.

Features:
1. KL divergence calculation (feature distortion metric)
2. Convert KL divergence to weights (direct mapping)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

EPS = 1e-20


class FeatureHookCollector:
    def __init__(self, model, target_modules=None):
        self.model = model
        self.features = []
        self.hooks = []
        self.target_modules = target_modules

    def _create_hook(self):
        def hook(module, input, output):
            self.features.append(output.detach().clone())

        return hook

    def register_hooks(self):
        if self.target_modules is None:
            target_modules = self._find_weight_layers(self.model)
        else:
            target_modules = self.target_modules

        for module in target_modules:
            hook = module.register_forward_hook(self._create_hook())
            self.hooks.append(hook)

    def _find_weight_layers(self, model):
        target_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                target_layers.append(module)
        return target_layers

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear_features(self):
        self.features.clear()

    def get_features(self):
        return self.features

    def __enter__(self):
        self.clear_features()
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


def collect_layer_features(model, x):
    features = []

    def hook_fn(module, input, output):
        features.append(output.detach().clone())

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(x)

    for hook in hooks:
        hook.remove()

    return features


def compute_kl_divergence(features_clean, features_adv):
    """
    Compute per-layer KL divergence as uncertainty metric.

    Low KL (low distortion) -> layer is robust -> increase perturbation
    High KL (high distortion) -> layer is fragile -> reduce perturbation

    Args:
        features_clean: List of clean sample features
                       - For Conv layers: [batch, channels, H, W] (4D)
                       - For ViT: [batch, seq_len, hidden] (3D)
                       - For Linear: [batch, num_classes] (2D, logits)
        features_adv: List of adversarial sample features, same shape as features_clean

    Returns:
        kl_per_layer: OrderedDict {layer_idx: kl_value}
                     where layer_idx corresponds to weight layer index

    Note:
        - For Conv layers with spatial dimensions, apply global average pooling
        - For Linear layers, directly use logits without pooling
        - This ensures one KL value per weight parameter group
    """
    kl_per_layer = OrderedDict()

    if not features_clean or not features_adv:
        return kl_per_layer

    for i, (feat_clean, feat_adv) in enumerate(zip(features_clean, features_adv)):
        # Handle different feature dimensions
        if feat_clean.dim() == 4:  # Conv: [batch, channels, H, W]
            # Global average pooling to [batch, channels]
            feat_clean_pool = F.adaptive_avg_pool2d(feat_clean, 1).view(
                feat_clean.size(0), -1
            )
            feat_adv_pool = F.adaptive_avg_pool2d(feat_adv, 1).view(
                feat_adv.size(0), -1
            )
        elif feat_clean.dim() == 3:  # ViT: [batch, seq_len, hidden]
            # Average over sequence dimension
            feat_clean_pool = feat_clean.mean(dim=1)
            feat_adv_pool = feat_adv.mean(dim=1)
        elif feat_clean.dim() == 2:  # Linear: [batch, num_classes]
            # Use logits directly (no pooling needed)
            feat_clean_pool = feat_clean
            feat_adv_pool = feat_adv
        else:
            # Unexpected dimension, skip this layer
            continue

        # Convert to probability distributions via softmax
        p = F.softmax(feat_clean_pool, dim=1)
        q = F.softmax(feat_adv_pool, dim=1)

        # Compute KL divergence: KL(p || q)
        # where p is the probability distribution from clean features
        # and q is from adversarial features
        kl = F.kl_div(torch.log(q + EPS), p, reduction="batchmean")
        kl_per_layer[i] = kl.item()

    return kl_per_layer


def compute_layer_weights_from_kl(kl_per_layer, kl_sensitivity=1.0):
    """
    Convert KL divergence to layer weights using exponential mapping.

    Direct mapping: KL divergence -> weight
    - Low KL (robust layer) -> high weight (strong perturbation)
    - High KL (fragile layer) -> low weight (weak perturbation)

    Args:
        kl_per_layer: {layer_idx: kl_value} from compute_kl_divergence()
        kl_sensitivity: Sensitivity parameter controlling mapping strength
                       - Typical range: 0.5 - 2.0
                       - Higher value: sharper differentiation between layers

    Returns:
        layer_weights: {layer_idx: weight_value}
                      where weight_value in range [0, 1]

    Note:
        The exponential mapping ensures:
        - weight = exp(-sensitivity * normalized_KL)
        - This is a smooth, monotonic mapping
        - Default sensitivity=1.0 provides balanced weighting
    """
    layer_weights = OrderedDict()

    if not kl_per_layer:
        return layer_weights

    # Normalize KL values to [0, 1] range
    kl_values = list(kl_per_layer.values())
    kl_min, kl_max = min(kl_values), max(kl_values)

    if kl_max > kl_min:
        # Standard normalization
        kl_norm = {
            i: (kl - kl_min) / (kl_max - kl_min + EPS) for i, kl in kl_per_layer.items()
        }
    else:
        # All KL values are the same, use uniform weights
        kl_norm = {i: 0.5 for i in kl_per_layer.keys()}

    # Exponential mapping: weight = exp(-beta * kl_norm)
    # Lower KL (robust) -> higher weight
    # Higher KL (fragile) -> lower weight
    beta = kl_sensitivity
    for idx, kl_norm_val in kl_norm.items():
        weight = torch.exp(
            torch.tensor(-beta * kl_norm_val, dtype=torch.float32)
        ).item()
        layer_weights[idx] = weight

    return layer_weights


def compute_all_layer_weights(features_clean, features_adv, kl_sensitivity=1.0):
    """
    One-stop computation of layer weights from features.

    This is the main API for layer weight calculation.
    Direct mapping: features -> KL divergence -> weights

    Args:
        features_clean: List of clean sample features
        features_adv: List of adversarial sample features
        kl_sensitivity: KL sensitivity parameter (default 1.0)

    Returns:
        layer_weights: {layer_idx: weight_value}
                      Ordered from layer 0 to N-1

    Example:
        # Assuming model returns features when called with feats=True
        logits_clean, feats_clean = model(x_clean, feats=True)
        logits_adv, feats_adv = model(x_adv, feats=True)

        layer_weights = compute_all_layer_weights(feats_clean, feats_adv)
        # layer_weights: {0: 0.8, 1: 0.6, 2: 0.9, ..., 20: 0.5}

        uawp.perturb(diff_scale, layer_weights)
    """
    # Compute KL divergence for each layer
    kl_per_layer = compute_kl_divergence(features_clean, features_adv)

    # Convert KL to weights
    layer_weights = compute_layer_weights_from_kl(
        kl_per_layer, kl_sensitivity=kl_sensitivity
    )

    return layer_weights


def get_default_layer_weights(num_layers, default_weight=1.0):
    """
    Get default layer weights (uniform).

    Use this when features are not available.

    Args:
        num_layers: Number of weight layers
        default_weight: Default weight value (default 1.0 for no differentiation)

    Returns:
        layer_weights: {layer_idx: default_weight}
    """
    return OrderedDict({i: default_weight for i in range(num_layers)})


class LayerWeightTracker:
    def __init__(self):
        self.layer_weights_history = []
        self.epochs = []

    def record_epoch(self, epoch, layer_weights):
        if layer_weights is None:
            return

        self.epochs.append(epoch)
        weights_values = [
            layer_weights.get(i, 1.0) for i in sorted(layer_weights.keys())
        ]
        self.layer_weights_history.append(weights_values)

    def plot_heatmap(self, save_path=None, title="Layer Weights Heatmap"):
        if not self.layer_weights_history or not self.epochs:
            return

        heatmap_data = np.array(self.layer_weights_history).T

        plt.figure(figsize=(max(12, len(self.epochs) * 0.5), 8))
        sns.heatmap(
            heatmap_data,
            cmap="YlOrRd",
            cbar_kws={"label": "Weight Value"},
            xticklabels=self.epochs,
            yticklabels=[f"Layer {i}" for i in range(len(heatmap_data))],
        )
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Layer (Shallow → Deep)", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close()

    def get_history(self):
        return self.layer_weights_history, self.epochs
