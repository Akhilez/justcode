from utils.data_manager import DataManager
from utils.grapher import Grapher
from akipy.layers import Input, Dense
from akipy.neural_network import Sequential, Metrics

import os

# --------------PRE_PROCESSING-----------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DataManager.load_and_save_split_data(x_path='data/MNISTnumImages5000.txt', y_path='data/MNISTnumLabels5000.txt', data_set_name='Mnist', parent_dir='data')

# dm = DataManager(data_path='data/iris.data')

dm = DataManager(x_train_path='data/irisxTrain.npy', y_train_path='data/irisyTrain.npy', x_test_path='data/irisxTest.npy', y_test_path='data/irisyTest.npy')

# dm = DataManager(x_train_path='data/MnistxTrain.npy', y_train_path='data/MnistyTrain.npy', x_test_path='data/MnistxTest.npy', y_test_path='data/MnistyTest.npy')

dm.load(split=True, one_hot=True)

# ---------------MODEL DESIGN------------------------

model = Sequential()

model.add(Input(units=4))
model.add(Dense(units=10, activation='sigmoid'))
model.add(Dense(units=3, activation='sigmoid'))

model.compile(optimizer='SGD', loss='MSE', metrics=['error', 'every_tenth_hit_rate', 'every_tenth_classification_error'])

# ----------------TRAINING--------------------------------

metrics = model.train(dm.x_train, dm.y_train, validation_set=(dm.x_test, dm.y_test), epochs=151, lr=0.1, momentum=0.01)

# ---------------TESTING-------------------------------

y_pred = model.test(dm.x_test)
y_pred = Metrics.get_winner_take_all(y_pred)

# -------------------PLOTTING------------------------------

confusion_matrix = Metrics.get_confusion_matrix(dm.y_test, y_pred)
print(f'Confusion matrix: \n{confusion_matrix}')

Grapher.plot_generic(metrics.tenth_epoch_indices, metrics.tenth_epoch_hit_rates, "Epoch vs Hit-Rate", "Epochs",
                     "Hit-Rate", 'hit_rate')
Grapher.plot_generic(metrics.tenth_epoch_indices, metrics.tenth_epoch_classification_error, "Epoch vs Error", "Epochs",
                     "Error", 'error')

# ------------------SAVING MODEL--------------------------

# model.save('test', parent_dir='models')

# model = model.load('test', parent_dir='models')
