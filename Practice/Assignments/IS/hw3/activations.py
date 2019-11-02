from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    name = 'None'

    def __init__(self):
        pass

    @abstractmethod
    def f(self, x):
        pass

    @abstractmethod
    def f_derivative(self, x):
        pass


class LinearActivation(ActivationFunction):
    name = 'linear'

    def f(self, x):
        return x

    def f_derivative(self, x):
        return 1


class SigmoidActivation(ActivationFunction):
    name = 'sigmoid'

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def f_derivative(self, x):
        fx = self.f(x)
        return fx * (1 - fx)


def get_activation_function(name, **kwargs):
    if name == LinearActivation.name:
        return LinearActivation(**kwargs)
    elif name == SigmoidActivation.name:
        return SigmoidActivation(**kwargs)
    else:
        raise Exception(f'The activation function {name} not found.')
