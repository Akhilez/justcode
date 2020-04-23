# DL46.py CS5173/6073 cheng 2020
# This is the autoencoder with bdata1.csv 
# We display the codes for the inputs in a plot.
# We can see the denoising effect by showing the decoding results 
# turned to binary using the threshold 0.5.
# Usage: python DL46.py

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

data1 = pd.read_csv("bdata1.csv")
data2 = data1.to_numpy()
print(data2)
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

reconstructed = model1(data3).gt(0.5).int()
print(reconstructed)

