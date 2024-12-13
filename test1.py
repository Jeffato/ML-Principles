import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.io import decode_image

import io
from PIL import Image
import PIL.ImageOps
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_errors = []
train_errors = []
conf_mat = np.zeros((10, 10), dtype=np.int32)

# (idx of image, euclidean distance)
high_misclass = [(0, sys.maxsize), (0, sys.maxsize), (0, sys.maxsize), (0, sys.maxsize), (0, sys.maxsize),
                 (0, sys.maxsize), (0, sys.maxsize), (0, sys.maxsize), (0, sys.maxsize), (0, sys.maxsize)]

num_classes = 10
num_epochs = 20
# ----------------------Creating BitMaps----------------------#
RBFTensor = torch.load('RBF.pt')


# ----------------------model----------------------#
class LeNet5(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=0, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv5 = nn.Linear(400, 120)
        self.fc6 = nn.Linear(120, 84)
        self.initalize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = 1.7159 * torch.tanh(x * 2 / 3)
        x = self.conv3(x)
        x = 1.7159 * torch.tanh(x * 2 / 3)
        x = x.reshape(x.size(0), -1)
        x = self.conv5(x)
        x = self.fc6(x)
        squared_diff = torch.pow(x - RBFTensor, 2)
        result = torch.sum(squared_diff, dim=1)
        result = result.view(1, 10)
        return result

    def initalize_weights(self):
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                in_feat = sum(p.numel() for p in m.parameters())
                nn.init.uniform_(m.weight, a=-2.4 / in_feat, b=2.4 / in_feat)


# ----------------------Data----------------------#
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


# given test data in format: png and separate label text document. Want to import this data into the data set and evalue

class MNIST(Dataset):
    def __init__(self, split="train", transform=None):
        self.datapath = "./data/"
        self.split = split
        self.transform = transform
        with open("./data/" + self.split + "_label.txt", "r") as f:
            self.labels = f.readlines()
        f.close()

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        image = Image.open(self.datapath + self.split + "/" + str(idx) + ".png")
        image = self.transform(image)
        label = int(self.labels[idx][0])
        return image, label


def test(dataloader, model):
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        vals, predicted = torch.min(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        conf_mat[predicted[0].item(), labels[0].item()] += 1

        if predicted != labels:
            if high_misclass[predicted[0].item()][1] > vals[0].item():
                high_misclass[predicted[0].item()] = (images, vals[0].item())
    print("test accuracy:", correct / total)


def main():
    mnist_test = MNIST(split="Test_distort", transform=transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((32, 32)),
        transforms.ToTensor()]))
    test_dataloader = DataLoader(mnist_test, batch_size=1, shuffle=False)
    model = torch.load("LeNet.pt")
    test(test_dataloader, model)


# ----------------------Testing----------------------#
main()
print(conf_mat)

