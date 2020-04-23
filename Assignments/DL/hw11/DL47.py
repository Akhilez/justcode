# DL47.py CS5173/6073 cheng 2020
# This is the autoencoder with bdata1.csv 
# After training we will only use the encoder (model1[0])
# The user will see encoder's weight, a 2 x 10 matrix
# and asked to choose one of the two features (0 or 1)
# The program will start with a random noise input
# and run the encoder to get the code for the input
# then use gradient ascent to modify the input to increase the chosen feature
# After a few iterations, the random input will be changed to a
# "typical pattern" specified by the chosen feature.
# This is similar to style transfer from a pretrained network to 
# a random or given input.
# Usage: python DL47.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

data1 = pd.read_csv("bdata1.csv")
data2 = data1.to_numpy()
data3 = torch.tensor(data2, dtype=torch.float32)

model1 = nn.Sequential(
	nn.Linear(10, 2, bias=False),
	nn.Linear(2, 10, bias=False))

optimizer = optim.SGD(model1.parameters(), 1e-2, momentum=0.3, nesterov=True)
loss_fun = nn.MSELoss()
num_epochs = 5000

for epoch in range(1, num_epochs + 1):
	p = model1(data3)
	loss = loss_fun(p, data3)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	if epoch == 1 or epoch % 1000 == 0:
		print('Epoch {}, Loss {}'.format(epoch, float(loss)))

codes = model1[0](data3).data.T

plt.rcParams['figure.figsize'] = (3.5, 2.5)
plt.scatter(codes[0], codes[1])
plt.show()

print()
print(list(model1[0].parameters()))
print()

print('Enter the hidden node (0 or 1) whose feature you want to enhance')
input1 = int(input())

data4 = torch.tensor(np.random.random((1, 10)), dtype=torch.float32, requires_grad=True)
print(data4)
print(data4.gt(0.5).int())

for i in range(5):
    c = model1[0](data4)
    c[0][input1].backward()
    data4 = data4.add(data4.grad * 0.5).clone().detach().requires_grad_(True)
    print(data4.gt(0.5).int())
