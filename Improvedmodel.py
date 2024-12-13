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
num_epochs = 5
#----------------------model----------------------#

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

#----------------------Data----------------------#
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


# Create dataset instance
train = DigitDataset(df_train["image"],
                     df_train["label"],
                     transform=transforms.Compose([
                         transforms.Grayscale(1),
                         transforms.Resize((32, 32)),
                         transforms.ToTensor()
                     ]))

# load datasets
train_load = torch.utils.data.DataLoader(dataset=train,
                                         batch_size=1,
                                         shuffle=True)



#----------------------Training----------------------#
model = LeNet5(num_classes).to(device)

# Setting the loss function
cost = nn.CrossEntropyLoss()

# Setting the optimizer with the model parameters and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# this is defined to print how many steps are remaining when training
total_step = len(train_load)
model.train()
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch}")
    for i, (images, labels) in enumerate(train_load):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        loss = cost(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if (i + 1) % 1000 == 0:
        #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
        #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
torch.save(model,'LeNet2.pt')