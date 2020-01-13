import numpy as np

from akipy.metrics import Metrics
from utils.data_manager import DataManager
from utils.grapher import Grapher
from akipy.layers import Input, Dense
from akipy.neural_network import Sequential

import matplotlib.pyplot as plt

# --------------PRE_PROCESSING-----------------------

dm = DataManager(x_train_path='data/MnistxTrain.npy', y_train_path='data/MnistyTrain.npy', x_test_path='data/MnistxTest.npy', y_test_path='data/MnistyTest.npy')

dm.load(split=True, one_hot=True)

# ---------------MODEL DESIGN------------------------

encoder_model = Sequential.load('mnist_auto_encoder', 'models', find_latest=True)
encoder_hidden_layer = encoder_model.layers[1]
encoder_hidden_layer.lr = 0.000000001  # So that it will learn slower.

model = Sequential(name='mnist_fine_tuned_classifier')

model.add(Input(units=784))
model.add(encoder_hidden_layer)
model.add(Dense(units=10, activation='sigmoid'))

model.compile(optimizer='SGD', loss='MSE', metrics=['error', 'every_tenth_hit_rate', 'every_tenth_classification_error'])

# -----------------OR LOAD MODEL--------------------------

# model = Sequential.load('mnist_auto_encoder', 'models.bkp', find_latest=True)

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

# Plotting error
Grapher.plot_generic(metrics.tenth_epoch_indices, metrics.tenth_epoch_classification_error, "Fine-Tuned Encoder Classifier", "Epochs", "Error", 'fine_tuned_error')

# -----------------------------------------------------------------------
# Random 20 neuron plots
neuron_indices = np.random.randint(0, 100, (20,))
neuron_indices.sort()
random_neuron_weights = Dense.remove_bias(encoder_hidden_layer.weights[neuron_indices]).reshape((20, 28, 28))

encoder_model = Sequential.load('mnist_auto_encoder', 'models', find_latest=True)
random_encoder_weights = Dense.remove_bias(encoder_model.layers[1].weights[neuron_indices]).reshape((20, 28, 28))

fig, axs = plt.subplots(8, 5, num=Grapher.get_new_fig_number(), figsize=(10, 15), constrained_layout=True)
axs = axs.reshape((2, 20))

for i in range(20):
    img = random_neuron_weights[i]
    axs[0][i].imshow(img, cmap='gray')
    title = f'Fine-Tuned Neurons {neuron_indices[i]}' if i == 0 else str(neuron_indices[i])
    axs[0][i].set_title(title)

for i in range(20):
    img = random_encoder_weights[i]
    axs[1][i].imshow(img, cmap="gray")
    title = f'Encoder Neurons {neuron_indices[i]}' if i == 0 else str(neuron_indices[i])
    axs[1][i].set_title(title)

fig.suptitle('Features of hidden layer neurons in fine-tuned classifier and auto-encoder')
fig.savefig('figures/fine_tuned_vs_auto_encoder.png')
fig.show()
