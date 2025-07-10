import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config_and_utils import CONFIG

class CelebAVFL(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.data = datasets.CelebA(root=root, split=split, target_type="attr",
                                    transform=transform, download=True)
        self.target_idx = self.data.attr_names.index(CONFIG["target_attr"])
        self.privacy_idxs = [self.data.attr_names.index(attr) for attr in CONFIG["privacy_attrs"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, attr = self.data[idx]
        w = img.shape[2]
        left = img[:, :, :w//2]
        right = img[:, :, w//2:]

        x1 = left.flatten()
        x2 = right.flatten()

        target = (attr[self.target_idx].item() + 1) // 2
        priv1 = (attr[self.privacy_idxs[0]].item() + 1) // 2
        priv2 = (attr[self.privacy_idxs[1]].item() + 1) // 2

        return (x1, x2), target, (priv1, priv2)

def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_data = CelebAVFL(root="./data", split="train", transform=transform)
    test_data = CelebAVFL(root="./data", split="test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader
