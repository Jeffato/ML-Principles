import torch
import torch.nn as nn
import torchvision.transforms as transforms

import PIL.ImageOps
from PIL import Image
import os
import numpy as np






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
        avg = avg+img/len(images)

    avg = np.array(np.round(avg), dtype=np.int32)
    tensor = torch.from_numpy(avg)

    output = model(tensor)

    to_pil_image = transforms.ToPILImage()

    pil_image = to_pil_image(output)
    resize = pil_image.resize((7, 12))
    RBF = np.array(resize)
    RBFarrays.append(RBF)




