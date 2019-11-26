import numpy as np
import matplotlib.pyplot as plt

from akipy.layers import Input, Som2D
from akipy.losses import ReverseExponentialDecay
from akipy.neural_network import Sequential
from utils.data_manager import DataManager

# ------------------Preprocessing-----------------------------

np.random.seed(1)

dm = DataManager()
attributes = [
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # Dove
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Hen
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Duck
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1],  # Goose
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],  # Owl
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],  # Hawk
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],  # Eagle
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],  # Fox
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],  # Dog
    [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],  # Wolf
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],  # Cat
    [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],  # Tiger
    [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],  # Lion
    [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],  # Horse
    [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],  # Zebra
    [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Cow
]
animals = ['Dove', 'Hen', 'Duck', 'Goose', 'Owl', 'Hawk', 'Eagle', 'Fox', 'Dog', 'Wolf', 'Cat', 'Tiger', 'Lion',
           'Horse', 'Zebra', 'Cow']
train_set = []
test_set = []

for i in range(16):
    animal = [1 if j == i else 0 for j in range(16)]
    data_point = list(animal)
    data_point.extend(attributes[i])
    train_set.append(data_point)
    data_point = list(animal)
    data_point.extend([0 for j in range(13)])
    test_set.append(data_point)

dm.x_train = np.array(train_set)
dm.x_test = np.array(test_set)

test_attributes = [
 [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],  # deer
 [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # pig
 [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],  # wolverine
 [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # ostrich
 [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],  # bat
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # blue whale
 [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1],  # human
]
new_animals = ['Deer', 'Pig', 'Wolverine', 'Ostrich', 'Bat', 'Blue Whale', 'Human']

new_test_set = []
for i in range(len(test_attributes)):
    new_animal = [0 for j in range(16)]
    new_animal.extend(test_attributes[i])
    new_test_set.append(new_animal)
new_test_set = np.array(new_test_set)

# ---------------------------MODEL---------------------------------

model = Sequential('som_animal_clusters')
model.add(Input(units=29))
model.add(Som2D(units=(10, 10), sigma=2, time_constant=1000, decay=ReverseExponentialDecay(time_constant=500)))

model.compile(optimizer='SGD', loss='WTA-min', metrics=['error'])

# ----------------------- TRAINING AND TESTING ---------------------------

model.train(x_train=dm.x_train, y_train=dm.x_train, epochs=200, lr=0.1)

feature_maps = model.test(x_test=dm.x_test)

training_maps = model.test(x_test=dm.x_train)

new_animals_maps = model.test(x_test=new_test_set)

# -------------------------PLOTTING Feature Maps ------------------------

fig, axs = plt.subplots(4, 4, num=1, figsize=(7, 7), constrained_layout=True)
axs = axs.flatten()
for i in range(16):
    axs[i].imshow(training_maps[i], cmap="gray")
    axs[i].set_title(animals[i])

fig.suptitle('Training Feature Maps')
fig.savefig('figures/feature_maps.png')
fig.show()

# -----------------------PLOTTING Testing feature maps -----------------

fig, axs = plt.subplots(4, 4, num=2, figsize=(7, 7), constrained_layout=True)
axs = axs.flatten()
for i in range(16):
    axs[i].imshow(feature_maps[i], cmap="gray")
    axs[i].set_title(animals[i])

fig.suptitle('Testing Feature Maps')
fig.savefig('figures/feature_maps_test.png')
fig.show()

# ----------------------------PLOT 1 ------------------------------

highest_neuron_for_animals = []
for i in range(16):
    highest_neuron_for_animals.append(np.unravel_index(feature_maps[i].argmin(), feature_maps[i].shape))
print(highest_neuron_for_animals)
print()
labelled_highest_neurons = []
for i in range(10):
    row_neurons = []
    for j in range(10):
        animals_matched = []
        for animal_i in range(16):
            neuron_min = highest_neuron_for_animals[animal_i]
            if i == neuron_min[0] and j == neuron_min[1]:
                animals_matched.append(animals[animal_i])
        row_neurons.append('.'.join(animals_matched))
    labelled_highest_neurons.append(row_neurons)
for i in labelled_highest_neurons:
    print(i)
print()

# fig, axs = plt.subplots(1, 1, num=3, figsize=(14, 7), constrained_layout=True)
# axs.axis('off')
# axs.axis('tight')
# axs.table(labelled_highest_neurons, loc='center')
# axs.set_title("Winner neuron for each animal")
# fig.savefig('figures/winning_neuron_for_animals.png')
# fig.show()

# ------------------------------PLOT 2-------------------------------

all_neurons_responses = []
for i in range(10):
    neuron_row_responses = []
    for j in range(10):
        neuron_responses = []
        for animal_i in feature_maps:
            neuron_responses.append(animal_i[i][j])
        arg_min = neuron_responses.index(min(neuron_responses))
        neuron_row_responses.append(animals[arg_min])
    all_neurons_responses.append(neuron_row_responses)

for i in all_neurons_responses:
    print(i)

fig, axs = plt.subplots(1, 1, num=4, figsize=(7, 3), constrained_layout=True)
axs.axis('off')
axs.axis('tight')
axs.table(all_neurons_responses, loc='center')
axs.set_title("Most responsive animal for each neuron")
fig.savefig('figures/animal_for_each_neuron.png')
fig.show()

# ---------------------------NEW ANIMALS Feature Maps-----------------------------

fig, axs = plt.subplots(3, 3, num=2.5, figsize=(7, 7), constrained_layout=True)
axs = axs.flatten()
for i in range(7):
    axs[i].imshow(new_animals_maps[i], cmap="gray")
    axs[i].set_title(new_animals[i])

fig.suptitle('New Animals Feature Maps')
fig.savefig('figures/new_animals_feature_maps.png')
fig.show()

# ------------------------Table Map new animals------------------------

highest_neuron_for_animals = []
for i in range(7):
    highest_neuron_for_animals.append(np.unravel_index(new_animals_maps[i].argmin(), new_animals_maps[i].shape))
print(highest_neuron_for_animals)
print()
labelled_highest_neurons = []
for i in range(10):
    row_neurons = []
    for j in range(10):
        animals_matched = []
        for animal_i in range(7):
            neuron_min = highest_neuron_for_animals[animal_i]
            if i == neuron_min[0] and j == neuron_min[1]:
                animals_matched.append(new_animals[animal_i])
        row_neurons.append('.'.join(animals_matched))
    labelled_highest_neurons.append(row_neurons)
for i in labelled_highest_neurons:
    print(i)
print()

fig, axs = plt.subplots(1, 1, num=6, figsize=(7, 3), constrained_layout=True)
axs.axis('off')
axs.axis('tight')
axs.table(labelled_highest_neurons, loc='center')
axs.set_title("Winner neuron for each NEW animal")
fig.savefig('figures/winning_neuron_for_new_animals.png')
fig.show()

# ---------------------------Table of neurons animals 2------------------------

all_neurons_responses = []
for i in range(10):
    neuron_row_responses = []
    for j in range(10):
        neuron_responses = []
        for animal_i in new_animals_maps:
            neuron_responses.append(animal_i[i][j])
        arg_min = neuron_responses.index(min(neuron_responses))
        neuron_row_responses.append(new_animals[arg_min])
    all_neurons_responses.append(neuron_row_responses)

for i in all_neurons_responses:
    print(i)

fig, axs = plt.subplots(1, 1, num=7, figsize=(7, 3), constrained_layout=True)
axs.axis('off')
axs.axis('tight')
axs.table(all_neurons_responses, loc='center')
axs.set_title("Most responsive NEW animal for each neuron")
fig.savefig('figures/new_animal_for_each_neuron.png')
fig.show()
