import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import decode_image

import io
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DigitDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(io.BytesIO(self.data.iloc[idx]["bytes"]))
        label = self.labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# load data
splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])

# Create dataset instance
train = DigitDataset(df_train["image"],
                     df_train["label"],
                     transform=transforms.Compose([
                         transforms.Resize((32, 32)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                     ]))

test = DigitDataset(df_test["image"],
                    df_train["label"],
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
                    ]))
# load datasets
train_load = torch.utils.data.DataLoader(dataset=train,
                                         batch_size=1,
                                         shuffle=True)

test_load = torch.utils.data.DataLoader(dataset=test,
                                        batch_size=1,
                                        shuffle=True)
