import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler


def mnist(root, batch_size=64, num_workers=0):
    train_set = datasets.MNIST(root=root + 'mnist/',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = datasets.MNIST(root=root + 'mnist/',
                              train=False,
                              download=True,
                              transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def cifar10(root, batch_size=64, num_workers=0, ddp=False, world_size=None, rank=None):
    transform = transforms.ToTensor()
    train_set = datasets.CIFAR10(root=root + 'cifar10/',
                                 train=True,
                                 download=False,
                                 transform=transform)

    if ddp:
        sampler = DistributedSampler(train_set,
                                     num_replicas=world_size,
                                     rank=rank,
                                     shuffle=True,
                                     seed=42)

        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  sampler=sampler,
                                  pin_memory=True,
                                  drop_last=False)
    else:
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=False)

    test_set = datasets.CIFAR10(root=root + 'cifar10/',
                                train=False,
                                download=False,
                                transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              drop_last=False)

    return train_loader, test_loader


def cifar100(root, batch_size=64, num_workers=0):
    train_set = datasets.CIFAR100(root=root + 'cifar100/',
                                  train=True,
                                  download=True,
                                  transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = datasets.CIFAR100(root=root + 'cifar100/',
                                 train=False,
                                 download=True,
                                 transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def svhn(root, batch_size=64, num_workers=0):
    train_set = datasets.SVHN(root=root + 'svhn/',
                              split='train',
                              download=True,
                              transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = datasets.SVHN(root=root + 'svhn/',
                             split='test',
                             download=True,
                             transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


class CIFAR10Augmented:
    def __init__(self, root):
        self.cifar = datasets.CIFAR10
        self.directory = root + 'cifar10/'
        self.tensor = transforms.ToTensor()

    # 1. original image
    def original(self):
        train_set_original = self.cifar(root=self.directory, train=True, download=True, transform=self.tensor)
        return train_set_original

    # 2. horizontal flip and jitter
    def horizontal_flip_jitter(self):
        flip = transforms.RandomHorizontalFlip(p=0.5)
        jitter = transforms.ColorJitter(brightness=(.7, 1.3), contrast=(.7, 1.3), saturation=(.7, 1.3))
        transform = transforms.Compose([self.tensor, flip, jitter])
        train_set_flip_jitter = self.cifar(root=self.directory, train=True, download=False, transform=transform)
        return train_set_flip_jitter

    # 3. random crop and affine
    def random_crop_affine(self):
        crop = transforms.RandomResizedCrop(size=(32, 32), scale=(.7, 1.3))
        affine = transforms.RandomAffine(degrees=(-10, 10), translate=(0.0, 0.1), scale=(.9, 1.1))
        transform = transforms.Compose([self.tensor, crop, affine])
        train_set_crop_affine = self.cifar(root=self.directory, train=True, download=False, transform=transform)
        return train_set_crop_affine

    # 4. gaussian blur
    def gaussian_blur(self):
        blur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.01, 1.0))
        transform = transforms.Compose([self.tensor, blur])
        train_set_blur = self.cifar(root=self.directory, train=True, download=False, transform=transform)
        return train_set_blur


def cifar10_augmented(root, batch_size=64, num_workers=0):
    train_set = []
    augment = CIFAR10Augmented(root=root)

    train_set.append(augment.original())
    train_set.append(augment.horizontal_flip_jitter())
    train_set.append(augment.random_crop_affine())
    train_set.append(augment.gaussian_blur())

    train_set = torch.utils.data.ConcatDataset(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = datasets.CIFAR10(root=root + 'cifar10/', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


class CIFAR100Augmented:
    def __init__(self, root):
        self.cifar = datasets.CIFAR100
        self.directory = root + 'cifar100/'
        self.tensor = transforms.ToTensor()

    # 1. original image
    def original(self):
        train_set_original = self.cifar(root=self.directory, train=True, download=True, transform=self.tensor)
        return train_set_original

    # 2. horizontal flip and jitter
    def horizontal_flip_jitter(self):
        flip = transforms.RandomHorizontalFlip(p=0.25)
        crop = transforms.RandomResizedCrop(size=(32, 32), scale=(.5, 1.))
        jitter = transforms.ColorJitter(brightness=(.7, 1.3), contrast=(.7, 1.3), saturation=(.7, 1.3))
        transform = transforms.Compose([self.tensor, flip, crop, jitter])
        train_set_flip_jitter = self.cifar(root=self.directory, train=True, download=False, transform=transform)
        return train_set_flip_jitter

    # 3. affine transformation
    def affine_transformation_blur(self):
        affine = transforms.RandomAffine(degrees=(-90, 90), translate=(0.0, .25), scale=(.9, 1.1))
        blur = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 1.5))
        transform = transforms.Compose([self.tensor, affine, blur])
        train_set_blur = self.cifar(root=self.directory, train=True, download=False, transform=transform)
        return train_set_blur


def cifar100_augmented(root, batch_size=64, num_workers=0):
    train_set = []
    augment = CIFAR100Augmented(root=root)

    train_set.append(augment.original())
    train_set.append(augment.horizontal_flip_jitter())
    train_set.append(augment.affine_transformation_blur())

    train_set = torch.utils.data.ConcatDataset(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = datasets.CIFAR100(root=root + 'cifar100/',
                                 train=False,
                                 download=True,
                                 transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt


    def imshow(img, name, n):
        img = np.transpose(img.numpy(), (1, 2, 0))
        plt.subplot(1, 2, n)
        plt.title(name)
        plt.imshow(img)


    aug = CIFAR10Augmented(root='./data/')

    train = torch.utils.data.DataLoader(aug.original(), batch_size=128, shuffle=False, num_workers=0)
    x_train1, _ = next(iter(train))

    train = torch.utils.data.DataLoader(aug.horizontal_flip_jitter(), batch_size=128, shuffle=False, num_workers=0)
    x_train2, _ = next(iter(train))

    for i1, i2 in zip(x_train1, x_train2):
        plt.figure('figure')
        imshow(i1, name='original image', n=1)
        imshow(i2, name='augmented image', n=2)
        plt.show()
        break
