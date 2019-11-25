from abc import ABC, abstractmethod
import numpy as np
from akipy.activations import get_activation_function
import math


class Layer(ABC):
    name = None
    _next_layer_number = 1

    def __init__(self, units, activation='linear', lr=None):
        self.n_units = units
        self.activation = get_activation_function(activation)
        self.lr = lr
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

    def __init__(self, units, activation='linear', lr=None):
        super().__init__(units, activation, lr=lr)
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
        delta_w = np.outer(delta, self.prev_xq) * (lr if self.lr is None else self.lr)
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


class Som2D(Layer):
    name = 'input'

    def __init__(self, units, lr=None, decay=None, **kwargs):
        super().__init__(units, lr=lr, **kwargs)
        self.r = None
        self.weights = None
        self.prev_yh = None
        self.prev_xq = None
        self.prev_min = None
        from akipy.losses import GaussianRateDecay
        self.decay = decay if decay is None else GaussianRateDecay()

    def feed(self, xq, **kwargs):
        distance_map = self.get_distance_map(xq)
        self.prev_yh = distance_map
        self.prev_xq = xq
        self.prev_min = np.unravel_index(distance_map.argmin(), distance_map.shape)
        return self.prev_yh

    def init_weights(self, weights=None):
        if weights is None:
            size = list(self.n_units)
            size.append(self.input_size)
            self.weights = np.random.uniform(low=-0.5, high=0.5, size=tuple(size))
        else:
            self.weights = np.array(weights)

    def back_propagate(self, lr, error, epoch=None, **kwargs):
        (n_rows, n_cols) = self.n_units
        for row_i in range(n_rows):
            for col_j in range(n_cols):
                lr = lr if self.lr is None else self.lr
                neuron_distance = self.euclidean_distance((row_i, col_j), self.prev_min)
                decay = self.decay.decay(neuron_distance ** 2, epoch)
                error = (self.prev_xq[col_j] - self.weights[row_i][col_j])

                change = lr * decay * error
                self.weights[row_i][col_j] += change

        return self.prev_yh

    def set_input_size(self, input_size):
        super().set_input_size(input_size)
        self.init_weights()

    @staticmethod
    def euclidean_distance(x1, x2):
        square_sum = 0
        for i in range(len(x1)):
            square_sum += (x1[i] - x2[i]) ** 2
        return square_sum ** 0.5

    def get_serialized_weights(self):
        return self.weights.tolist()

    def get_distance_map(self, xq):
        distance_map = np.zeros(self.n_units)
        (n_rows, n_cols) = self.n_units
        for row_i in range(n_rows):
            for col_j in range(n_cols):
                distance = self.euclidean_distance(self.weights[row_i][col_j], xq)
                distance_map[row_i][col_j] = distance
        return distance_map

    def create_output_map(self, min_indices):
        zeros = np.zeros(shape=self.n_units)
        zeros[min_indices[0]][min_indices[1]] = 1
        return zeros


def create_layer_from_structure(layer):
    activation = layer['activation']
    n_units = layer['n_units']
    name = layer['name']
    if name == Dense.name:
        layer_obj = Dense(units=n_units, activation=activation)
    elif name == Input.name:
        layer_obj = Input(units=n_units, activation=activation)
    elif name == Som2D.name:
        # TODO: Return Som
        layer_obj = Som2D()
    else:
        raise Exception(f'The layer {layer["name"]} cannot be created.')
    layer_obj.init_weights(layer['weights'])
    return layer_obj
