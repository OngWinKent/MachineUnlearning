from torchvision import transforms, datasets
from torchvision.datasets import CIFAR100, CIFAR10, MNIST, FashionMNIST
from torch.utils.data import Dataset


CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


transform_train_augment = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]

transform_test = [
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
]


class MNist(MNIST):
    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
        augment: bool= True,
        img_size: int= 28
    ):

        transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y


class FMNist(FashionMNIST):
    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
        augment: bool = True,
        img_size: int = 28
    ):
        transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y


class Cifar10(CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
        augment: bool = True,
        img_size: int = 32
    ):
        if train:
            if augment:
                transform = transform_train_augment
            else:
                transform = transform_test
        else:
            transform = transform_test
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y


class Cifar100(CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
        augment: bool = True,
        img_size: int = 32
    ):
        if train:
            if augment:
                transform = transform_train_augment
            else:
                transform = transform_test
        else:
            transform = transform_test
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y


class Cifar20(CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
        augment: bool = True,
        img_size: int = 32
):
        if train:
            if augment:
                transform = transform_train_augment
            else:
                transform = transform_test
        else:
            transform = transform_test
        transform.append(transforms.Resize(img_size))
        transform = transforms.Compose(transform)

        super().__init__(root=root, train=train, download=download, transform=transform)

        # This map is for the matching of subclases to the superclasses. E.g., rocket (69) to Vehicle2 (19:)
        # Taken from https://github.com/vikram2000b/bad-teaching-unlearning
        self.coarse_map = {
            0: [4, 30, 55, 72, 95],
            1: [1, 32, 67, 73, 91],
            2: [54, 62, 70, 82, 92],
            3: [9, 10, 16, 28, 61],
            4: [0, 51, 53, 57, 83],
            5: [22, 39, 40, 86, 87],
            6: [5, 20, 25, 84, 94],
            7: [6, 7, 14, 18, 24],
            8: [3, 42, 43, 88, 97],
            9: [12, 17, 37, 68, 76],
            10: [23, 33, 49, 60, 71],
            11: [15, 19, 21, 31, 38],
            12: [34, 63, 64, 66, 75],
            13: [26, 45, 77, 79, 99],
            14: [2, 11, 35, 46, 98],
            15: [27, 29, 44, 78, 93],
            16: [36, 50, 65, 74, 80],
            17: [47, 52, 56, 59, 96],
            18: [8, 13, 48, 58, 90],
            19: [41, 69, 81, 85, 89],
        }

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        coarse_y = None
        for i in range(20):
            for j in self.coarse_map[i]:
                if y == j:
                    coarse_y = i
                    break
            if coarse_y != None:
                break
        if coarse_y == None:
            print(y)
            assert coarse_y != None
        return x, y, coarse_y


# https://github.com/facundoq/tinyimagenet/blob/main/tinyimagenet.py
class TinyImagenet(Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
        augment: bool = True,
        img_size: int = 64
    ):
        self.root = root
        self.train = train
        self.augment= augment
        self.img_size = img_size
        self.data = self._prepare_data()

    def _prepare_data(self):
        if self.train:
            if self.augment:
                transform = transforms.Compose([
                    transforms.Resize(self.img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                ])
            dataset_dir = f"{self.root}/tiny-imagenet-200/train"
        else:
            transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
            dataset_dir = f"{self.root}/tiny-imagenet-200/preprocessed_val"

        dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y