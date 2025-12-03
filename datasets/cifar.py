import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2



def get_cifar10_loaders(config):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if config.config == "cons_at":
        transform_train = MultiDataTransform(transform_train)
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=config.data_dir,
        train=False,
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


def get_cifar100_loaders(config):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if config.config == "cons_at":
        transform_train = MultiDataTransform(transform_train)
    train_dataset = torchvision.datasets.CIFAR100(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=config.data_dir,
        train=False,
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
