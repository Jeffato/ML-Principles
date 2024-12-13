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
conf_mat = np.zeros((10,10),dtype = np.int32)

# (idx of image, euclidean distance)
high_misclass = [(0,sys.maxsize),(0,sys.maxsize),(0,sys.maxsize),(0,sys.maxsize),(0,sys.maxsize),(0,sys.maxsize),(0,sys.maxsize),(0,sys.maxsize),(0,sys.maxsize),(0,sys.maxsize)]


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


# load data
splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])

# Create dataset instance
train = DigitDataset(df_train["image"],
                     df_train["label"],
                     transform=transforms.Compose([
                         transforms.Grayscale(1),
                         transforms.Resize((32, 32)),
                         transforms.ToTensor()
                     ]))

test = DigitDataset(df_test["image"],
                    df_test["label"],
                    transform=transforms.Compose([
                        transforms.Grayscale(1),
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()
                    ]))
# load datasets
train_load = torch.utils.data.DataLoader(dataset=train,
                                         batch_size=1,
                                         shuffle=True)

test_load = torch.utils.data.DataLoader(dataset=test)

# ----------------------Training----------------------#

# TODO: Loss function,gradient, and update

class LeNetLoss(nn.Module):
    def __init__(self):
        super(LeNetLoss, self).__init__()

    def forward(self, predictions, target):
        mask = target[0].item()
        losscalc = predictions[0, mask] + torch.log(
            torch.pow(torch.tensor(torch.e), -0.1) + torch.sum(torch.exp(torch.neg(predictions))) - torch.exp(
                torch.neg(predictions[0][mask])))
        return losscalc


model = LeNet5(num_classes).to(device)

cost = LeNetLoss()

# this is defined to print how many steps are remaining when training
total_step = len(train_load)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_load):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        loss = cost(outputs, labels)
        # Backward and optimize
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                new_val = p - 0.000001*p.grad
                p.copy_(new_val)
        if (i + 1) % 10000 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# ----------------------Testing----------------------#
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in train_load:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.min(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_errors.append((total-correct)/total)
        print("train errors:", train_errors)

        correct = 0
        total = 0

        for images, labels in test_load:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            vals, predicted = torch.min(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if epoch+1 == num_epochs:
                conf_mat[predicted[0].item(),labels[0].item()] += 1

                if predicted != labels:
                    if high_misclass[predicted[0].item()][1] > vals[0].item():
                        high_misclass[predicted[0].item()] = (images,vals[0].item())



        test_errors.append((total-correct)/total)
        print("errors:", test_errors)
#---------------------------Evaluation-----------------------#
print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
print(conf_mat)

TensorToImage = transforms.ToPILImage()

for i in range(10):
    img = TensorToImage(high_misclass[i][0][0])
    img.show()
    img.save(str(i) + '.png')



plt.plot(train_errors, label = "train")
plt.plot(test_errors, label = "test")

plt.xlabel("iterations")
plt.ylabel("error")

plt.show()
plt.savefig('error.png')

torch.save(model, 'LeNet.pt')
