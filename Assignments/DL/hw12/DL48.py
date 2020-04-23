# DL48.py CS5173/6073 cheng 2020
# This is a variant and simplification of DL40.py (Assignment 10)
# The content image is replaced with a white noise (randomly generated) input_img
# vgg19 is replaced with inception_v3's first five conv2d layers as "model"
# The output of model(input_img) has 192 channels, each with a feature map
# The channel with the highest mean feature map value is selected as the "channel"
# The negative of the mean of the feature map for the channel ("channelvalue") is the loss function
# An LBFGS optimizer is used to find gradient on the input_img and 20 iterations is performed in each step
# The channel mean is displayed after each optimizer.step() and you can see it is increasing
# after a number of steps (DL40.py uses 300 steps), the white noise input_img is transfered so that
# a specific feature learned by the pretrained inception_v3 shows up.
# inception_v3 has many more upper layers and channels there may be more interesting
# Usage: python DL48.py

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

inception3 = torchvision.models.inception_v3(pretrained=True)
input_img = torch.randn([1, 3, 128, 128])
unloader = transforms.ToPILImage()

model = nn.Sequential()

iter = inception3.children()
for i in range(5):
    model.add_module('layer{}'.format(i), next(next(iter).children()))

output = model(input_img)
a, b, c, d = output.size()
m = torch.mean(output.view(a * b, c * d), 1)
channel = m.argmax()
print(channel, m[channel])

optimizer = optim.LBFGS([input_img.requires_grad_()])

for s in range(10):
    
    def closure():
        input_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        output = model(input_img)
        channelvalue = -torch.mean(output[0][channel])
        channelvalue.backward()
        return channelvalue

    optimizer.step(closure)
    print(s, torch.mean(model(input_img)[0][channel]))

plt.figure()
plt.imshow(unloader(input_img.squeeze(0)))
plt.title('channel {}'.format(channel))
plt.show()




