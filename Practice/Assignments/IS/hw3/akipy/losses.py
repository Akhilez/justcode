from abc import ABC, abstractmethod
import math


class LossFunction(ABC):
    @abstractmethod
    def f(self, yq, yh):
        pass

    @abstractmethod
    def f_derivative(self, yq, yh):
        pass


class MseLoss(LossFunction):
    name = 'MSE'

    def f(self, yq, yh):
        return (yq - yh) ** 2

    def f_derivative(self, yq, yh):
        return yq - yh


def get_loss_function(name, **kwargs):
    if name == MseLoss.name:
        return MseLoss(**kwargs)
    else:
        raise Exception(f"Loss function {name} is not found.")


class RateDecay(ABC):
    @abstractmethod
    def decay(self, value, time_i, **kwargs):
        pass


class GaussianRateDecay(RateDecay):
    def __init__(self, sigma=0.5, time_constant=1000):
        self.sigma = sigma
        self.time_constant = time_constant

    def decay(self, value, time_i, **kwargs):
        sigma_square = 2 * (self.sigma * math.exp(-1 * time_i / self.time_constant)) ** 2
        return math.exp(-1 * value / sigma_square)
