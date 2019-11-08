import numpy as np

from utils.data_manager import DataManager
from utils.grapher import Grapher
from akipy.layers import Input, Dense
from akipy.neural_network import Sequential

import matplotlib.pyplot as plt

# --------------PRE_PROCESSING-----------------------

# DataManager.load_and_save_split_data(x_path='data/MNISTnumImages5000.txt', y_path='data/MNISTnumLabels5000.txt', data_set_name='Mnist', parent_dir='data')

# dm = DataManager(data_path='data/iris.data')

# dm = DataManager(x_train_path='data/irisxTrain.npy', y_train_path='data/irisyTrain.npy', x_test_path='data/irisxTest.npy', y_test_path='data/irisyTest.npy')

dm = DataManager(x_train_path='data/MnistxTrain.npy', y_train_path='data/MnistyTrain.npy', x_test_path='data/MnistxTest.npy', y_test_path='data/MnistyTest.npy')

dm.load(split=True, one_hot=True)

# ---------------MODEL DESIGN------------------------

model = Sequential(name='mnist_auto_encoder')

model.add(Input(units=784))
model.add(Dense(units=100, activation='sigmoid'))
model.add(Dense(units=784, activation='sigmoid'))

model.compile(optimizer='SGD', loss='MSE', metrics=['error', 'iter_errors', 'every_tenth_error'])

# -----------------OR LOAD MODEL--------------------------

model = Sequential.load('mnist_auto_encoder', 'models.bkp', find_latest=True)

# ----------------TRAINING--------------------------------

metrics = model.train(dm.x_train, dm.x_train, validation_set=(dm.x_test, dm.x_test), epochs=11, lr=0.1, momentum=0.1)
model.save(parent_dir='models')

# ---------------TESTING-------------------------------

x_test_pred, x_test_metrics = model.test(dm.x_test, dm.x_test)
x_train_pred, x_train_metrics = model.test(dm.x_train, dm.x_train)

# -------------------PLOTTING------------------------------

# Plotting error
Grapher.plot_generic(metrics.tenth_epoch_indices, metrics.tenth_epoch_errors, "Auto Encoder: Epoch vs Error", "Epochs", "Error", 'auto_encoder_error')

# Get errors
x_train_error = x_train_metrics.errors[-1]
x_test_error = x_test_metrics.errors[-1]

# Plot with training and testing error (bars)
fig, axs = Grapher.create_figure(1, 1, figsize=(4, 4))
axs.set_title('Train and test errors')
axs.bar(['Train', 'Test'], [x_train_error, x_test_error], align='center')
axs.set_ylabel('Error')
fig.savefig('figures/auto_encoder_bars_1.png')
fig.show()

# -----------------------------------------------------------------------
# Get errors for each class
x_classes = []
x_train_errors = []
x_test_errors = []
classes = np.unique(dm.y_test, axis=0)
for class_i in classes:
    x_train_error = 0
    x_test_error = 0
    for i in range(len(dm.y_train)):
        if all(dm.y_train[i] == class_i):
            x_train_error += x_train_metrics.iter_errors[0][i]

    for i in range(len(dm.y_test)):
        if all(dm.y_test[i] == class_i):
            x_test_error += x_test_metrics.iter_errors[0][i]
    x_train_errors.append(x_train_error)
    x_test_errors.append(x_test_error)
    x_classes.append(class_i.argmax())

# Plot errors for each class
fig, axs = Grapher.create_figure(1, 1, figsize=(7, 5))
axs.set_title('Errors for each class')
axs.bar(x_classes, [x_train_errors[i] for i in range(len(classes))], label='Train', width=0.35)
axs.bar([class_i + 0.35 for class_i in x_classes], [x_test_errors[i] for i in range(len(x_classes))], label='Test', width=0.35)
axs.set_xticks(x_classes)
axs.set_xlabel('Classes')
axs.legend(['Train', 'Test'])
fig.savefig('figures/auto_encoder_bars_2.png')
fig.show()

# -----------------------------------------------------------------------
# Random 20 neuron plots
neuron_indices = np.random.randint(0, 100, (20,))
neuron_indices.sort()
random_neuron_weights = Dense.remove_bias(model.layers[1].weights[neuron_indices]).reshape((20, 28, 28))

classification_model = Sequential.load('mnist_classification', 'models', find_latest=True)
random_classification_weights = Dense.remove_bias(classification_model.layers[1].weights[neuron_indices]).reshape((20, 28, 28))

fig, axs = plt.subplots(8, 5, num=Grapher.get_new_fig_number(), figsize=(10, 15), constrained_layout=True)
axs = axs.reshape((2, 20))

for i in range(20):
    img = random_neuron_weights[i]
    axs[0][i].imshow(img, cmap='gray')
    title = f'Encoder Neurons {neuron_indices[i]}' if i == 0 else str(neuron_indices[i])
    axs[0][i].set_title(title)

for i in range(20):
    img = random_classification_weights[i]
    axs[1][i].imshow(img, cmap="gray")
    title = f'Classifier Neurons {neuron_indices[i]}' if i == 0 else str(neuron_indices[i])
    axs[1][i].set_title(title)

fig.suptitle('Features of hidden layer neurons in auto-encoder and classifier')
fig.savefig('figures/auto_encoder_vs_classifier_images.png')
fig.show()

# -----------------------------------------------------------------------
# Plotting output for 8 test points
random_test_indices = np.random.randint(0, len(dm.x_test), 8)
random_test_indices.sort()

random_x_test_pred = x_test_pred[random_test_indices].reshape((8, 28, 28))
random_x_test = dm.x_test[random_test_indices].reshape((8, 28, 28))

fig, axs = plt.subplots(2, 8, num=Grapher.get_new_fig_number(), figsize=(10, 3), constrained_layout=True)
fig.suptitle('Encoder: Original vs Predicted Images')

for i in range(8):
    axs[0][i].imshow(random_x_test[i], cmap='gray')
    title = f'Original Images {random_test_indices[i]}' if i == 0 else str(random_test_indices[i])
    axs[0][i].set_title(title)

for i in range(8):
    axs[1][i].imshow(random_x_test_pred[i], cmap='gray')
    title = f'Predicted Images {random_test_indices[i]}' if i == 0 else str(random_test_indices[i])
    axs[1][i].set_title(title)

fig.savefig('figures/auto_encoder_real_vs_predicted_images.png')
fig.show()
