# DL2.py CS5173/6073 2020 cheng
# A multilayer perceptron for MNIST
# prints loss in training and accuracy in testing
# Usage: at command prompt, run "python DL2.py"

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

num_inputs, num_outputs, num_hiddens = 784, 10, 256
batch_size_train = 256
n_epochs = 10

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/', train=True, download=True,
                               transform=torchvision.transforms.ToTensor()),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/', train=False, download=True,
                               transform=torchvision.transforms.ToTensor()))
test_size = 10000  # may be derived from test_loader

model2 = nn.Sequential(
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs))

optimizer = optim.SGD(model2.parameters(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(n_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.reshape(-1, num_inputs)
        p = model2(data)
        train_loss = loss_fn(p, target)
        if batch_idx % 100 == 0:
            print('train', epoch, batch_idx, float(train_loss))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    m = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.reshape(-1, num_inputs)
        if int(torch.argmax(model2(data))) == int(target[0]):
            m = m + 1
    print("test", epoch, m, "among", test_size, "correctly classified")
