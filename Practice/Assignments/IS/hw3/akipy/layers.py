from abc import ABC, abstractmethod
import numpy as np
from akipy.activations import get_activation_function


class Layer(ABC):
    name = None
    _next_layer_number = 1

    def __init__(self, units, activation='linear'):
        self.n_units = units
        self.activation = get_activation_function(activation)
        self.input_size = None
        self.weights = None
        self.layer_name = f'{self.name}_layer_{self.get_next_layer_number()}'

    def set_input_size(self, input_size):
        self.input_size = input_size

    @abstractmethod
    def feed(self, xq, **kwargs):
        pass

    @abstractmethod
    def init_weights(self, weights=None):
        pass

    @abstractmethod
    def back_propagate(self, lr, error, **kwargs):
        pass

    def get_serialized(self):
        return {
            'name': self.name,
            'n_units': self.n_units,
            'activation': self.activation.name,
            'weights': self.get_serialized_weights(),
        }

    @abstractmethod
    def get_serialized_weights(self):
        pass

    @staticmethod
    def get_next_layer_number():
        next_layer_no = Layer._next_layer_number
        Layer._next_layer_number += 1
        return next_layer_no


class Dense(Layer):
    name = 'dense'

    def __init__(self, units, activation='linear'):
        super().__init__(units, activation)
        self.weights = None
        self.prev_xq = None
        self._prev_s = None
        self._prev_weight_change = None

    def feed(self, xq, **kwargs):
        xq = self.get_augmented_x(xq)
        s = self.weights.dot(xq)
        h = self.activation.f(s)

        self.prev_xq = xq
        self._prev_s = s

        return h

    def back_propagate(self, lr, error, momentum=None, **kwargs):
        delta = error * self.activation.f_derivative(self._prev_s)
        delta_w = np.outer(delta, self.prev_xq) * lr
        next_delta = delta.dot(self.remove_bias(self.weights))

        if momentum is not None and self._prev_weight_change is not None:
            delta_w += momentum * self._prev_weight_change
        self._prev_weight_change = delta_w

        self.weights += delta_w

        return next_delta

    def get_serialized_weights(self):
        return self.weights.tolist()

    def init_weights(self, weights=None):
        if weights is None:
            self.weights = np.random.uniform(low=-0.5, high=0.5, size=(self.n_units, self.input_size + 1))
        else:
            self.weights = np.array(weights)

    @staticmethod
    def remove_bias(weights):
        return weights[:, 1:]

    @staticmethod
    def get_augmented_x(xq):
        x_aug = np.ones(len(xq) + 1)
        x_aug[1:] = xq
        return x_aug

    def set_input_size(self, input_size):
        super().set_input_size(input_size)
        self.init_weights()


class Input(Layer):
    name = 'input'

    def __init__(self, units, activation='linear'):
        super().__init__(units, activation)
        self.x = None
        self.weights = None

    def feed(self, xq, **kwargs):
        return xq

    def back_propagate(self, lr, error, **kwargs):
        return error

    def set_input(self, x):
        if len(x) != self.n_units:
            raise Exception(f"Input length {x.shape} does not match with {self.n_units}.")
        self.x = x

    def get_serialized_weights(self):
        return ()

    def init_weights(self, weights=None):
        pass


def create_layer_from_structure(layer):
    activation = layer['activation']
    n_units = layer['n_units']
    name = layer['name']
    if name == Dense.name:
        layer_obj = Dense(units=n_units, activation=activation)
    elif name == Input.name:
        layer_obj = Input(units=n_units, activation=activation)
    else:
        raise Exception(f'The layer {layer["name"]} cannot be created.')
    layer_obj.init_weights(layer['weights'])
    return layer_obj
