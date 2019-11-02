from abc import ABC, abstractmethod
import random


class Optimizer(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def get_data_point(self, **kwargs):
        pass

    @abstractmethod
    def feed(self, **kwargs):
        pass


class StochasticGradientDescent(Optimizer):
    name = 'SGD'

    def __init__(self, model):
        super().__init__(model)
        self.x_train = None
        self.y_train = None
        self._not_seen = None
        self.x_test = None
        self.y_test = None
        self.lr = None
        self.deltas = []

    def set_training_data(self, x_train, y_train, lr, validation_set=None):
        self.x_train = x_train
        self.y_train = y_train
        self._not_seen = [i for i in range(len(x_train))]
        self.x_test, self.y_test = validation_set if validation_set is not None else (None, None)
        self.lr = lr

    def feed(self, xq, yq, **kwargs):
        hl = xq
        for layer in self.model.layers:
            hl = layer.feed(hl)

        self.back_propagate(xq, yq, hl)

    def back_propagate(self, xq, yq, yh):
        error = self.model.loss_function.f_derivative(yq, yh)
        for layer in self.model.layers.__reversed__():
            error = layer.back_propagate(self.lr, error)

    def get_data_point(self):
        random_index = random.randint(0, len(self._not_seen))
        not_seen_index = self._not_seen[random_index]
        del self._not_seen[random_index]

        x = self.x_train[not_seen_index]
        y = self.y_train[not_seen_index]

        return x, y


def get_optimizer(name, **kwargs):
    if name == StochasticGradientDescent.name:
        return StochasticGradientDescent(**kwargs)
    else:
        raise Exception(f'Optimizer {name} not found.')
