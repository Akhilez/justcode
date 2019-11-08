from utils.data_manager import DataManager
from utils.grapher import Grapher
from akipy.layers import Input, Dense
from akipy.neural_network import Sequential, Metrics

# --------------PRE_PROCESSING-----------------------

# DataManager.load_and_save_split_data(x_path='data/MNISTnumImages5000.txt', y_path='data/MNISTnumLabels5000.txt', data_set_name='Mnist', parent_dir='data')

# dm = DataManager(data_path='data/iris.data')

# dm = DataManager(x_train_path='data/irisxTrain.npy', y_train_path='data/irisyTrain.npy', x_test_path='data/irisxTest.npy', y_test_path='data/irisyTest.npy')

dm = DataManager(x_train_path='data/MnistxTrain.npy', y_train_path='data/MnistyTrain.npy', x_test_path='data/MnistxTest.npy', y_test_path='data/MnistyTest.npy')

dm.load(split=True, one_hot=True)

# ---------------MODEL DESIGN------------------------

model = Sequential(name='mnist_classification')

model.add(Input(units=784))
model.add(Dense(units=100, activation='sigmoid'))
model.add(Dense(units=10, activation='sigmoid'))

model.compile(optimizer='SGD', loss='MSE', metrics=['error', 'every_tenth_hit_rate', 'every_tenth_classification_error'])

# -----------------OR LOAD MODEL--------------------------

# model = Sequential.load('mnist_classification', 'models', find_latest=True)

# ----------------TRAINING--------------------------------

metrics = model.train(dm.x_train, dm.y_train, validation_set=(dm.x_test, dm.y_test), epochs=41, lr=0.5, momentum=0.1)
model.save(parent_dir='models')

# ---------------TESTING-------------------------------

y_pred = model.test(dm.x_test)
y_pred = Metrics.get_winner_take_all(y_pred)

y_pred_train = model.test(dm.x_train)
y_pred_train = Metrics.get_winner_take_all(y_pred_train)

# -------------------PLOTTING------------------------------

confusion_matrix = Metrics.get_confusion_matrix(dm.y_test, y_pred)
print(f'Confusion matrix for test set: \n{confusion_matrix}')

confusion_matrix = Metrics.get_confusion_matrix(dm.y_train, y_pred_train)
print(f'Confusion matrix for train set: \n{confusion_matrix}')

Grapher.plot_generic(metrics.tenth_epoch_indices, metrics.tenth_epoch_classification_error, "Classifier: Epoch vs Error", "Epochs", "Error", 'classification_error')
