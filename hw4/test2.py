import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.io import decode_image
import torch.nn.functional as F
import torch.cuda

import io
from PIL import Image

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

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
RBFTensor = RBFTensor.to(device)

# ----------------------model----------------------#

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=0, stride=1),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv5 = nn.Linear(400,120)
        self.fc6 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, num_classes)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=6),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(90, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 90)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)

        x = self.conv1(x)
        x = self.conv3(x)
        x = x.reshape(x.size(0), -1)
        x = self.conv5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        return x
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
    def __init__(self, split="train_distort", transform=None):
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
        vals, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        conf_mat[predicted[0].item(), labels[0].item()] += 1

        if predicted != labels:
            if high_misclass[predicted[0].item()][1] > vals[0].item():
                high_misclass[predicted[0].item()] = (images, vals[0].item())
    print("test accuracy:", correct / total)


def main():
    print("Loading Data...")

    mnist_test = MNIST(split="Test_distort", transform=transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((32, 32)),
        transforms.ToTensor()]))
    test_dataloader = DataLoader(mnist_test, batch_size=1, shuffle=False)

    print("Loading Model...")
    model = torch.load("LeNet5_2.pt")
    model = model.to(device)
    
    print("Testing Model...")
    test(test_dataloader, model)


# ----------------------Testing----------------------#
main()
print(conf_mat)