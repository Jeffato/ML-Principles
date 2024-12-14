import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import v2

import PIL.ImageOps
from PIL import Image
import os
import numpy as np
import random

path = os.getcwd()
newpath = os.path.join(path, 'data', 'test')
savepath = os.path.join(path,'data','Test_distort2')
folder_dir = os.chdir(newpath)

perspective_transformer = v2.RandomPerspective(distortion_scale=0.7, p=1.0)
rotater = v2.RandomRotation(degrees=(0, 180))
for images in os.listdir(folder_dir):
    int = random.randint(0,1)
    img = Image.open(images)
    if int == 1:
        img = perspective_transformer(img)
        os.chdir(savepath)
        img.save(images)
        os.chdir(newpath)

    if int == 0:
        img = rotater(img)
        os.chdir(savepath)
        img.save(images)
        os.chdir(newpath)
