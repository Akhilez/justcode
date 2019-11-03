from abc import ABC, abstractmethod
import numpy as np
from activations import get_activation_function


class Layer(ABC):
    name = None

    def __init__(self, units, activation='linear'):
        self.n_units = units
        self.activation = get_activation_function(activation)
        self.input_size = None
        self.weights = None

    def set_input_size(self, input_size):
        self.input_size = input_size

    @abstractmethod
    def feed(self, xq, **kwargs):
        pass

    @abstractmethod
    def back_propagate(self, lr, error, **kwargs):
        pass

    def get_serialized(self):
        return {
            'name': self.name,
            'n_units': self.n_units,
            'activation': self.activation.name,
            'weights': self.weights.tolist(),
        }


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
        self.weights = np.random.random((self.n_units, input_size + 1))


class Input(Layer):
    name = 'input'

    def __init__(self, units, activation='linear'):
        super().__init__(units, activation)
        self.x = None
        self.weights = np.array([])

    def feed(self, xq, **kwargs):
        return xq

    def back_propagate(self, lr, error, **kwargs):
        # TODO: Decide what to do
        return error

    def set_input(self, x):
        if len(x) != self.n_units:
            raise Exception(f"Input length {x.shape} does not match with {self.n_units}.")
        self.x = x


def create_layer_from_structure(layer):
    # TODO: Return layer object from structure.
    return None
