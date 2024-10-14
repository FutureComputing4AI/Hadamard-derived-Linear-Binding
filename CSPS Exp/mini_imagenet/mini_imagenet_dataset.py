import torch
from torchvision import datasets, transforms


class AUGMENT:
    def __init__(self, directory, dataset, size=84):
        self.size = size
        self.dataset = dataset
        self.directory = directory
        self.tensor = transforms.ToTensor()

    def original(self):
        transform = transforms.Compose([transforms.Resize((self.size, self.size)), self.tensor])
        dataset = self.dataset(root=self.directory, transform=transform)

        torch.manual_seed(0)
        train_set, _ = torch.utils.data.random_split(dataset, [50_000, 10_000])
        return train_set

    # 1. crop + jitter + transform
    def augment_1(self):
        crop = transforms.RandomResizedCrop(size=(self.size, self.size), scale=(0.5, 1.0))
        jitter = transforms.ColorJitter(brightness=(.5, 1.5), contrast=(.5, 1.5), saturation=(.5, 1.5))
        transform = transforms.Compose([transforms.Resize((self.size, self.size)), crop, jitter, self.tensor])
        dataset = self.dataset(root=self.directory, transform=transform)

        torch.manual_seed(0)
        train_set_augment, _ = torch.utils.data.random_split(dataset, [50_000, 10_000])
        return train_set_augment

    # 2. affine + transform
    def augment_2(self):
        affine = transforms.RandomAffine(degrees=(-60, 60), translate=(0.0, .25), scale=(.9, 1.1))
        transform = transforms.Compose([transforms.Resize((self.size, self.size)), affine, self.tensor])
        dataset = self.dataset(root=self.directory, transform=transform)

        torch.manual_seed(0)
        train_set_augment, _ = torch.utils.data.random_split(dataset, [50_000, 10_000])
        return train_set_augment


def mini_imagenet(root, batch_size=64, num_workers=0, size=84):
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=root + 'mini_imagenet/', transform=transform)

    torch.manual_seed(0)
    train_set, test_set = torch.utils.data.random_split(dataset, [50_000, 10_000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def mini_imagenet_augment_84(root, batch_size=64, num_workers=0):
    train_set = []
    augment = AUGMENT(directory=root + 'mini_imagenet/', dataset=datasets.ImageFolder, size=84)

    train_set.append(augment.original())
    train_set.append(augment.augment_1())
    train_set.append(augment.augment_2())

    train_set = torch.utils.data.ConcatDataset(train_set)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    transform = transforms.Compose([transforms.Resize((84, 84)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=root + 'mini_imagenet/', transform=transform)
    torch.manual_seed(0)
    _, test_set = torch.utils.data.random_split(dataset, [50_000, 10_000])

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)

    return train_loader, test_loader


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt


    def imshow(img, name, n):
        img = np.transpose(img.numpy(), (1, 2, 0))
        plt.subplot(1, 2, n)
        plt.title(name)
        plt.imshow(img)


    train, test = mini_imagenet(root='../data/', batch_size=8)

    a = 0
    for x, y in train:
        a += x.shape[0]
    print(a)

    for i, (i1, i2) in enumerate(zip(train, test)):
        plt.figure('figure')
        imshow(i1[0][i], name='train image', n=1)
        imshow(i2[0][i], name='test image', n=2)
        print(i1[0].shape)
        print(i1[1][i])
        print(i2[1][i])
        plt.show()
