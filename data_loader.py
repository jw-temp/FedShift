from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms
import random
from fedlab.utils.dataset.partition import (
    CIFAR10Partitioner,
    CIFAR100Partitioner,
    FMNISTPartitioner,
)


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        # print("asda",self.tensors[0].shape)

        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = x.numpy()
            x = (x * 255).astype(np.uint8)
            x = Image.fromarray((x * 255).astype(np.uint8))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def get_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def load_cifar10_data(
    batch_size,
    num_clients,
    iid=True,
    shard_size=250,
    root="./data",
    dirichlet=False,
    alpha=0.5,
    seed=42,
):
    transform = get_transform()

    trainset = datasets.CIFAR10(root=root, train=True, download=True)
    testset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    if not dirichlet:
        if iid:
            indices = np.random.permutation(len(trainset))
            shuffled_data = np.array(trainset.data)[indices]
            shuffled_labels = torch.Tensor(trainset.targets)[indices]
            shard_size = len(trainset) // num_clients
            shard_inputs = list(torch.split(torch.Tensor(shuffled_data), shard_size))
            shard_labels = list(torch.split(torch.Tensor(shuffled_labels), shard_size))

            train_datasets = [
                CustomTensorDataset((inputs, labels), transform=transform)
                for inputs, labels in zip(shard_inputs, shard_labels)
            ]

            trainloaders = [
                DataLoader(dataset, batch_size=batch_size, shuffle=True)
                for dataset in train_datasets
            ]
        else:
            # Sort data based on labels
            sorted_indices = torch.argsort(torch.Tensor(trainset.targets))
            sorted_data = np.array(trainset.data)[sorted_indices]
            sorted_labels = torch.Tensor(trainset.targets)[sorted_indices]

            shard_size = len(sorted_labels) // num_clients
            # Split data into shards
            shard_inputs = list(torch.split(torch.Tensor(sorted_data), shard_size))
            shard_labels = list(torch.split(torch.Tensor(sorted_labels), shard_size))
            # shard_inputs = [shard.permute(0, 3, 1, 2) for shard in shard_inputs]  # Transposing the data
            # Separate shards by their label

            train_set_size = len(trainset.targets)
            num_of_labels = len(set(trainset.targets))
            train_datasets = []

            # Allocate two consecutive shards to each client

            for i in range(len(shard_inputs)):
                idx_1 = i
                idx_2 = (
                    idx_1 + 2 * (train_set_size // shard_size // num_of_labels)
                ) % len(shard_inputs)

                inputs = torch.cat([shard_inputs[idx_1], shard_inputs[idx_2]])
                labels = torch.cat([shard_labels[idx_1], shard_labels[idx_2]]).long()

                train_datasets.append(
                    CustomTensorDataset((inputs, labels), transform=transform)
                )

            even_pairs = [
                dataset for dataset in train_datasets if dataset[0][1].item() % 2 == 0
            ]
            odd_pairs = [
                dataset for dataset in train_datasets if dataset[0][1].item() % 2 == 1
            ]
            result = []
            random.shuffle(even_pairs)
            random.shuffle(odd_pairs)

            for e, o in zip(even_pairs, odd_pairs):
                result.append(e)
                result.append(o)
            train_datasets = result

            trainloaders = [
                DataLoader(dataset, batch_size=batch_size, shuffle=True)
                for dataset in train_datasets
            ]
    else:
        hetero_dir_part = CIFAR10Partitioner(
            trainset.targets,
            num_clients,
            balance=None,
            partition="dirichlet",
            dir_alpha=alpha,
            seed=seed,
        )

        # Create custom datasets for each client
        train_datasets = [
            CustomTensorDataset(
                (
                    torch.Tensor(np.array(trainset.data)[indices]),
                    torch.Tensor(torch.Tensor(trainset.targets)[indices]),
                ),
                transform=transform,
            )
            for _, indices in hetero_dir_part.client_dict.items()
        ]

        # Create data loaders for each client
        trainloaders = [
            DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for dataset in train_datasets
        ]

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloaders, testloader


def load_cifar100_data(
    batch_size,
    num_clients,
    iid=True,
    shard_size=250,
    root="./data",
    dirichlet=False,
    alpha=0.5,
    seed=42,
):
    transform = get_transform()

    trainset = datasets.CIFAR100(root=root, train=True, download=True)
    testset = datasets.CIFAR100(
        root=root,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    if not dirichlet:
        if iid:
            indices = np.random.permutation(len(trainset))
            shuffled_data = np.array(trainset.data)[indices]
            shuffled_labels = torch.Tensor(trainset.targets)[indices]
            shard_size = len(trainset) // num_clients
            shard_inputs = list(torch.split(torch.Tensor(shuffled_data), shard_size))
            shard_labels = list(torch.split(torch.Tensor(shuffled_labels), shard_size))

            train_datasets = [
                CustomTensorDataset((inputs, labels), transform=transform)
                for inputs, labels in zip(shard_inputs, shard_labels)
            ]

            trainloaders = [
                DataLoader(dataset, batch_size=batch_size, shuffle=True)
                for dataset in train_datasets
            ]
        else:
            # Sort data based on labels
            sorted_indices = torch.argsort(torch.Tensor(trainset.targets))
            sorted_data = np.array(trainset.data)[sorted_indices]
            sorted_labels = torch.Tensor(trainset.targets)[sorted_indices]

            shard_size = len(sorted_labels) // num_clients
            # Split data into shards
            shard_inputs = list(torch.split(torch.Tensor(sorted_data), shard_size))
            shard_labels = list(torch.split(torch.Tensor(sorted_labels), shard_size))
            # shard_inputs = [shard.permute(0, 3, 1, 2) for shard in shard_inputs]  # Transposing the data
            # Separate shards by their label

            train_set_size = len(trainset.targets)
            num_of_labels = len(set(trainset.targets))
            train_datasets = []

            # Allocate two consecutive shards to each client

            for i in range(len(shard_inputs)):
                idx_1 = i
                idx_2 = (
                    idx_1 + 2 * (train_set_size // shard_size // num_of_labels)
                ) % len(shard_inputs)

                inputs = torch.cat([shard_inputs[idx_1], shard_inputs[idx_2]])
                labels = torch.cat([shard_labels[idx_1], shard_labels[idx_2]]).long()

                train_datasets.append(
                    CustomTensorDataset((inputs, labels), transform=transform)
                )

            even_pairs = [
                dataset for dataset in train_datasets if dataset[0][1].item() % 2 == 0
            ]
            odd_pairs = [
                dataset for dataset in train_datasets if dataset[0][1].item() % 2 == 1
            ]
            result = []
            random.shuffle(even_pairs)
            random.shuffle(odd_pairs)

            for e, o in zip(even_pairs, odd_pairs):
                result.append(e)
                result.append(o)
            train_datasets = result

            trainloaders = [
                DataLoader(dataset, batch_size=batch_size, shuffle=True)
                for dataset in train_datasets
            ]
    else:
        hetero_dir_part = CIFAR100Partitioner(
            trainset.targets,
            num_clients,
            balance=None,
            partition="dirichlet",
            dir_alpha=alpha,
            seed=seed,
        )

        # Create custom datasets for each client
        train_datasets = [
            CustomTensorDataset(
                (
                    torch.Tensor(np.array(trainset.data)[indices]),
                    torch.Tensor(torch.Tensor(trainset.targets)[indices]),
                ),
                transform=transform,
            )
            for _, indices in hetero_dir_part.client_dict.items()
        ]

        # Create data loaders for each client
        trainloaders = [
            DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for dataset in train_datasets
        ]

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloaders, testloader


def load_FMNIST_data(
    batch_size,
    num_clients,
    iid=True,
    shard_size=250,
    root="./data",
    dirichlet=False,
    alpha=0.5,
    seed=42,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
        ]
    )
    trainset = datasets.FashionMNIST(root=root, train=True, download=True)
    testset = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )

    hetero_dir_part = FMNISTPartitioner(
        trainset.targets,
        num_clients,
        partition="noniid-#label",
        major_classes_num=2,
        seed=seed,
    )
    # Create custom datasets for each client
    train_datasets = [
        CustomTensorDataset(
            (
                torch.Tensor(np.array(trainset.data)[indices]),
                torch.Tensor(torch.Tensor(trainset.targets)[indices]),
            ),
            transform=transform,
        )
        for _, indices in hetero_dir_part.client_dict.items()
    ]

    # Create data loaders for each client
    trainloaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for dataset in train_datasets
    ]

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloaders, testloader
