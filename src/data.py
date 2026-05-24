import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


DATASET_CONFIGS = {
    "cifar10": {
        "dataset": torchvision.datasets.CIFAR10,
        "num_classes": 10,
    },
    "cifar100": {
        "dataset": torchvision.datasets.CIFAR100,
        "num_classes": 100,
    },
}


def normalize_dataset_name(dataset_name):
    return dataset_name.lower().replace("-", "")


def get_num_classes(dataset_name):
    dataset_key = normalize_dataset_name(dataset_name)
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"不支持的数据集：{dataset_name}")
    return DATASET_CONFIGS[dataset_key]["num_classes"]


def build_subset(dataset, size, seed):
    if size <= 0 or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_dataset_loaders(
    dataset_name,
    data_dir,
    train_size,
    test_size,
    batch_size,
    seed,
    num_workers=0,
    pin_memory=False,
):
    dataset_key = normalize_dataset_name(dataset_name)
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"不支持的数据集：{dataset_name}")

    dataset_cls = DATASET_CONFIGS[dataset_key]["dataset"]
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = dataset_cls(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    testset = dataset_cls(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    train_subset = build_subset(trainset, train_size, seed)
    test_subset = build_subset(testset, test_size, seed + 1)
    persistent_workers = num_workers > 0
    trainloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    testloader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return trainloader, testloader


def build_cifar10_loaders(data_dir, train_size, test_size, batch_size, seed, num_workers=0, pin_memory=False):
    return build_dataset_loaders(
        "cifar10",
        data_dir,
        train_size,
        test_size,
        batch_size,
        seed,
        num_workers,
        pin_memory,
    )
