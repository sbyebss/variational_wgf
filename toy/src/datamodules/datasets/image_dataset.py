import os
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
# from .celeba import CelebA


def mnist_dataset(data_path, img_size=32):
    train = MNIST(
        data_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        ),
    )
    return train


def get_img_dataset(config):  # pylint: disable=too-many-branches
    if config.dataset == "MNIST":
        return mnist_dataset(config.path, config.image_size)
    if config.random_flip is False:
        tran_transform = transforms.Compose(
            [transforms.Resize(config.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

    if config.dataset == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(config.path, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )

    elif config.dataset == "CELEBA":
        if config.random_flip:
            dataset = CelebA(
                root=os.path.join(config.path, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        transforms.CenterCrop(140),
                        transforms.Resize(config.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(config.path, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        transforms.CenterCrop(140),
                        transforms.Resize(config.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )

    return dataset
