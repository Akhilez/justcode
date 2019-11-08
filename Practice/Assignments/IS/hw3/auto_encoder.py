from utils.data_manager import DataManager
from utils.grapher import Grapher
from akipy.layers import Input, Dense
from akipy.neural_network import Sequential, Metrics

import matplotlib.pyplot as plt

# --------------PRE_PROCESSING-----------------------

# DataManager.load_and_save_split_data(x_path='data/MNISTnumImages5000.txt', y_path='data/MNISTnumLabels5000.txt', data_set_name='Mnist', parent_dir='data')

# dm = DataManager(data_path='data/iris.data')

dm = DataManager(x_train_path='data/irisxTrain.npy', y_train_path='data/irisyTrain.npy', x_test_path='data/irisxTest.npy', y_test_path='data/irisyTest.npy')

# dm = DataManager(x_train_path='data/MnistxTrain.npy', y_train_path='data/MnistyTrain.npy', x_test_path='data/MnistxTest.npy', y_test_path='data/MnistyTest.npy')

dm.load(split=True, one_hot=True)

# ---------------MODEL DESIGN------------------------

model = Sequential(name='iris_auto_encoder')

model.add(Input(units=4))
model.add(Dense(units=10, activation='sigmoid'))
model.add(Dense(units=4, activation='linear'))

model.compile(optimizer='SGD', loss='MSE', metrics=['error'])

# -----------------OR LOAD MODEL--------------------------

model = Sequential.load('iris_auto_encoder', 'models', find_latest=True)

# ----------------TRAINING--------------------------------

# metrics = model.train(dm.x_train, dm.x_train, validation_set=(dm.x_test, dm.x_test), epochs=51, lr=0.01, momentum=0.01)
# model.save(parent_dir='models')

# ---------------TESTING-------------------------------

x_test_pred, x_test_metrics = model.test(dm.x_test, dm.x_test)
x_train_pred, x_train_metrics = model.test(dm.x_train, dm.x_train)

# -------------------PLOTTING------------------------------
'''
# Get errors
x_train_error = x_train_metrics.errors[-1]
x_test_error = x_test_metrics.errors[-1]

# Plot with training and testing error (bars)
fig, axs = Grapher.create_figure(1, 1, figsize=(5, 5))
axs.set_title('Train and test errors')
axs.bar(['Train', 'Test'], [x_train_error, x_test_error], align='center')
axs.set_ylabel('Error')
fig.show()
'''

# Get errors for each class
x_train_errors = []
x_test_errors = []

# Plot errors for each class
fig, axs = Grapher.create_figure(2, 5)
fig.suptitle('Errors for each class')
axs = axs.flatten()
for i in range(len(axs)):
    axs.bar(['Train', 'Test'], [x_train_errors[i], x_test_errors[i]], align='center')
fig.show()
