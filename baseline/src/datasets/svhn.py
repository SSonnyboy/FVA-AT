import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2

def get_svhn_loaders(config):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.1),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if config.config == "cons_at":
        transform_train = MultiDataTransform(transform_train)
    train_dataset = torchvision.datasets.SVHN(
        root=config.data_dir,
        split='train',
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.SVHN(
        root=config.data_dir,
        split='test',
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    return train_loader, test_loader
