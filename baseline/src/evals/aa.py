import torch
from tqdm import tqdm


def evaluate_aa(model, device, test_loader, norm='Linf', eps=8/255, verbose=False):
    """
    使用AutoAttack评估模型的对抗鲁棒性

    AutoAttack 是最强的对抗攻击之一，包含以下攻击方法：
    - PGD (Projected Gradient Descent)
    - FGSM (Fast Gradient Sign Method)
    - C&W (Carlini & Wagner)
    - AutoFool

    Args:
        model: 要评估的模型
        device: 计算设备
        test_loader: 测试数据加载器
        norm: 范数类型，'Linf' (L-infinity), 'L2', 'L1'
        eps: 扰动限制
        verbose: 是否打印详细信息

    Returns:
        对抗精度（百分比）
    """
    try:
        import autoattack
    except ImportError:
        raise ImportError(
            "AutoAttack not installed. Install with:\n"
            "  pip install autoattack"
        )

    model.eval()

    # 收集所有图像和标签
    all_images = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())

    x_test = torch.cat(all_images).to(device)
    y_test = torch.cat(all_labels).to(device)

    if verbose:
        print(f"Running AutoAttack evaluation...")
        print(f"  Norm: {norm}")
        print(f"  Epsilon: {eps:.4f}")
        print(f"  Total samples: {len(x_test)}")

    # 创建 AutoAttack 对象
    adversary = autoattack.AutoAttack(
        model,
        norm=norm,
        eps=eps,
        version='standard'  # 标准版本（推荐用于评估）
    )

    # 运行攻击
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)

    # 计算鲁棒精度
    with torch.no_grad():
        logits = model(x_adv)
        _, predicted = logits.max(1)
        correct = predicted.eq(y_test).sum().item()
        total = len(y_test)

    robust_accuracy = 100.0 * correct / total

    if verbose:
        print(f"AutoAttack Robust Accuracy: {robust_accuracy:.2f}%")

    return robust_accuracy
