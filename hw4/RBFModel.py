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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 4
#----------------------Creating BitMaps----------------------#

#DOWLOAD digits from https://www.kaggle.com/datasets/karnikakapoor/digits?resource=download-directory
RBFarrays=[]
model = nn.MaxPool2d(8, 8)
path = os.getcwd()
for i in range(0,10):
    newpath = os.path.join(path, 'digits updated', str(i))
    os.chdir(newpath)
    files = os.listdir(os.getcwd())

    images = [filename for filename in files]
    width, height = Image.open(images[0]).size

    avg = np.zeros((1,height,width), dtype= np.float32)

    for image in images:
        image = Image.open(image).convert('L')
        img = np.array(PIL.ImageOps.invert(image))
        img= img.reshape((1,img.shape[0],img.shape[1]))
        tensor = torch.from_numpy(img)
        output = model(tensor)
        numpy = tensor.numpy()
        avg = avg+ numpy/len(images)

    avg = np.array(np.round(avg), dtype=np.int32)


    pil_image = Image.fromarray(avg[0])

    resize = pil_image.resize((7, 12))
    resize = resize.convert('L')
    RBF = np.array(resize)
    RBF = RBF.flatten()
    RBFarrays.append(RBF)

RBFarrays = np.array(RBFarrays)
RBFTensor = torch.from_numpy(RBFarrays)
#----------------------model----------------------#
class CustomTanh(nn.Module):
    def __init__(self):
        super(CustomTanh, self).__init__()

    def forward(self, x):
        return 1.7159 * torch.tanh(x * 2 / 3)

class CustomRBF(nn.Module):
    def __init__(self):
        super(CustomRBF, self).__init__()

    def forward(self,x):
        print(x)
        squared_diff = torch.pow(x - RBFTensor, 2)
        result = torch.sum(squared_diff, dim=1)
        result = result.view(1,10)
        return result

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=0, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            CustomTanh()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            CustomTanh()
        )
        self.conv5 = nn.Linear(400,120)
        self.fc6 = nn.Linear(120, 84)
        self.rbf = CustomRBF()

        self.initalize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv3(x)
        x = x.reshape(x.size(0), -1)
        x = self.conv5(x)
        x = self.fc6(x)
        x = self.rbf(x)
        return x

    def initalize_weights(self):
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                in_feat = sum(p.numel() for p in m.parameters())
                nn.init.uniform_(m.weight, a=-2.4/in_feat , b=2.4/in_feat)

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

test_load = torch.utils.data.DataLoader(dataset=test,
                                        batch_size=1,
                                        shuffle=True)


#----------------------Training----------------------#

#TODO: Lost function,gradient, and update

model = LeNet5(num_classes).to(device)


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_load:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
