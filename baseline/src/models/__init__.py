from .resnet import ResNet18
from .preactresnet import PreActResNet18
from .wideresnet import WideResNet


def get_model(config):
    """
    Load model based on config.model

    Supported models:
    - resnet18: ResNet18
    - wrn28_10: WideResNet (depth=28, width=10)
    - wrn34_10: WideResNet (depth=34, width=10)
    - preactresnet18: PreActResNet18
    """
    model_name = config.model.lower()

    if model_name == 'resnet18':
        return ResNet18(num_classes=config.num_classes)
    elif model_name == 'preactresnet18':
        return PreActResNet18(num_classes=config.num_classes)
    elif model_name == 'wrn28_10':
        return WideResNet(depth=28, widen_factor=10, num_classes=config.num_classes, dropRate=0.0)
    elif model_name == 'wrn34_10':
        return WideResNet(depth=34, widen_factor=10, num_classes=config.num_classes, dropRate=0.0)
    else:
        raise ValueError(f"Unknown model: {config.model}. Supported: resnet18, wrn28_10, wrn34_10, preactresnet18")
