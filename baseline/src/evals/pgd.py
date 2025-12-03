import torch
import torch.nn as nn
from collections import OrderedDict
from attacks import PGD


def evaluate_pgd(model, device, test_loader, eps, alpha, steps):
    """
    使用 PGD 攻击评估模型的鲁棒性

    Args:
        model: 模型
        device: 计算设备
        test_loader: 测试数据加载器
        eps: 扰动大小
        alpha: PGD 步长
        steps: PGD 步数

    Returns:
        accuracy: 鲁棒准确率（百分比）
    """
    model.eval()
    pgd_attack = PGD(eps=eps, alpha=alpha, steps=steps, loss_fn=nn.CrossEntropyLoss())

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images_adv = pgd_attack(model, images, labels)

            logits = model(images_adv)
            _, predicted = logits.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def evaluate_pgd_10(model, device, test_loader):
    eps = 8 / 255
    alpha = 2 / 255
    steps = 10
    return evaluate_pgd(model, device, test_loader, eps, alpha, steps)


def evaluate_pgd_20(model, device, test_loader):
    eps = 8 / 255
    alpha = 2 / 255
    steps = 20
    return evaluate_pgd(model, device, test_loader, eps, alpha, steps)

def evaluate_pgd_50(model, device, test_loader):
    eps = 8 / 255
    alpha = 2 / 255
    steps = 50
    return evaluate_pgd(model, device, test_loader, eps, alpha, steps)


def evaluate_pgd_classwise(model, device, test_loader, eps=8/255, alpha=2/255, steps=10):
    model.eval()
    pgd_attack = PGD(eps=eps, alpha=alpha, steps=steps, loss_fn=nn.CrossEntropyLoss())

    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images_adv = pgd_attack(model, images, labels)

            logits = model(images_adv)
            _, predicted = logits.max(1)

            for pred, label in zip(predicted, labels):
                label_idx = label.item()
                pred_idx = pred.item()

                if label_idx not in class_total:
                    class_total[label_idx] = 0
                    class_correct[label_idx] = 0

                class_total[label_idx] += 1
                if pred_idx == label_idx:
                    class_correct[label_idx] += 1

    class_accuracy = OrderedDict()
    for class_idx in sorted(class_total.keys()):
        if class_total[class_idx] > 0:
            class_accuracy[class_idx] = class_correct[class_idx] / class_total[class_idx]
        else:
            class_accuracy[class_idx] = 0.0

    return class_accuracy


def evaluate_pgd_10_classwise(model, device, test_loader):
    eps = 8 / 255
    alpha = 2 / 255
    steps = 10
    return evaluate_pgd_classwise(model, device, test_loader, eps, alpha, steps)
