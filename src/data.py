import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def build_subset(dataset, size, seed):
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_cifar10_loaders(data_dir, train_size, test_size, batch_size, seed):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    train_subset = build_subset(trainset, train_size, seed)
    test_subset = build_subset(testset, test_size, seed + 1)
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader
