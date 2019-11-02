from abc import ABC, abstractmethod
import numpy as np
from activations import get_activation_function


class Layer(ABC):
    def __init__(self, units, activation='linear'):
        self.n_units = units
        self.activation = get_activation_function(activation)
        self.input_size = None

    def set_input_size(self, input_size):
        self.input_size = input_size

    def get_serialized(self):
        # TODO: Get serialized layer
        pass


class Dense(Layer):
    def __init__(self, units, activation='linear'):
        super().__init__(units, activation)
        self.weights = None

    def set_input_size(self, input_size):
        super().set_input_size(input_size)
        self.weights = np.zeros((self.n_units, input_size))


class Input(Layer):
    def __init__(self, units, activation='linear'):
        super().__init__(units, activation)
        self.x = None

    def set_input(self, x):
        if len(x) != self.n_units:
            raise Exception(f"Input length {x.shape} does not match with {self.n_units}.")
        self.x = x


def create_layer_from_structure(layer):
    # TODO: Return layer object from structure.
    return None
