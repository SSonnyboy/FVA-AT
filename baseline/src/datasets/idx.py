import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image

class SVHN_idx(datasets.SVHN):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(SVHN_idx, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

        # unify the interface
        if not hasattr(self, 'data'):       # torch <= 0.4.1
            if self.train:
                self.data, self.targets = self.train_data, self.train_labels
            else:
                self.data, self.targets = self.test_data, self.test_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    @property
    def num_classes(self):
        return 10


class CIFAR10_idx(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10_idx, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

        # unify the interface
        if not hasattr(self, 'data'):       # torch <= 0.4.1
            if self.train:
                self.data, self.targets = self.train_data, self.train_labels
            else:
                self.data, self.targets = self.test_data, self.test_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    @property
    def num_classes(self):
        return 10


class CIFAR100_idx(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100_idx, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

        # unify the interface
        if not hasattr(self, 'data'):       # torch <= 0.4.1
            if self.train:
                self.data, self.targets = self.train_data, self.train_labels
            else:
                self.data, self.targets = self.test_data, self.test_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    @property
    def num_classes(self):
        return 100


def get_cifar10_loaders_idx(config):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = CIFAR10_idx(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = CIFAR10_idx(
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


def get_cifar100_loaders_idx(config):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = CIFAR100_idx(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = CIFAR100_idx(
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

def get_svhn_loaders_idx(config):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = SVHN_idx(
        root=config.data_dir,
        split='train',
        download=True,
        transform=transform_train
    )

    test_dataset = SVHN_idx(
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