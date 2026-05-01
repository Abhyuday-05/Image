# dataset.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import *

transform = transforms.Compose([
    transforms.Grayscale(input_channels),
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class CustomDataset(datasets.ImageFolder):
    def __init__(self, root):
        super().__init__(root, transform=transform)
        self.remap_labels()

    def remap_labels(self):
        new_samples = []
        new_targets = []

        for path, _ in self.samples:
            class_name = path.split(os.sep)[-2]
            label = 0 if class_name == 'LoPt' else 1

            new_samples.append((path, label))
            new_targets.append(label)

        self.samples = new_samples
        self.imgs = new_samples
        self.targets = new_targets


def get_dataloader(train=True):
    if train:
        dataset = CustomDataset(train_dir)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = CustomDataset(test_dir)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
