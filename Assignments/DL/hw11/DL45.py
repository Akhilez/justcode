# DL45.py CS5173/6073 cheng 2020
# implementing GAN according to 16.1 of d2l
# but using bdata1.csv as the "real data" seen only by the discriminator net_D
# the generator net_G tries to modify its weight so that its randomly generated 
# samples may go through the discriminator
# Usage: python DL45.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

data1 = pd.read_csv("bdata1.csv")
data2 = data1.to_numpy()
real_X = torch.tensor(data2, dtype=torch.float32)
print(real_X.gt(0.5).int())
print()

batch_size, latent_dim, n = 10, 2, 10
net_G = nn.Sequential(
    nn.Linear(latent_dim, n))

Z = torch.tensor(np.random.normal(0, 1, size=(batch_size, latent_dim)), 
                 dtype=torch.float32)
fake_X = net_G(Z)
print(Z)
print(fake_X)
print(fake_X.gt(0.5).int())

net_D = nn.Sequential(
    nn.Linear(n, latent_dim),
    nn.ReLU(),
    nn.Linear(latent_dim, 1),
    nn.Sigmoid())

ones = torch.ones((batch_size,1))
zeros = torch.zeros((batch_size,1))

lr_D, lr_G, num_epochs = 0.05, 0.01, 100
trainer_D = optim.Adam(net_D.parameters(), lr_D)
trainer_G = optim.Adam(net_G.parameters(), lr_G)
loss = nn.BCELoss()

def update_D():
    """Update discriminator"""
    real_Y = net_D(real_X)
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    trainer_D.zero_grad()
    loss_D.backward()
    trainer_D.step()
    return float(loss_D.sum())

def update_G(): 
    """Update generator"""
    # Recomputing fake_Y is needed since net_D is changed.
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones)
    trainer_G.zero_grad()
    loss_G.backward()
    trainer_G.step()
    return float(loss_G.sum())

for epoch in range(1, num_epochs+1):
    loss_G = update_G()
    loss_D = update_D()
    Z = torch.tensor(np.random.normal(0, 1, size=(batch_size, latent_dim)), 
                 dtype=torch.float32)
    if (epoch % 10 == 0):
        print('epoch', epoch)
        print(loss_G, loss_D)
        new_fake_X = net_G(Z)
        print('new fakes')
        print(new_fake_X.gt(0.5).int())
        print("discriminator's output to new fakes")
        print(net_D(new_fake_X))
