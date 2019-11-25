from abc import ABC, abstractmethod
import math

import numpy as np


class LossFunction(ABC):
    name = None

    @abstractmethod
    def f(self, yq, yh):
        pass

    @abstractmethod
    def f_derivative(self, yq, yh):
        pass


class MseLoss(LossFunction):
    name = 'MSE'

    def __init__(self, **kwargs):
        pass

    def f(self, yq, yh):
        return (yq - yh) ** 2

    def f_derivative(self, yq, yh):
        return yq - yh


class WinnerTakeAllLoss(LossFunction):
    name = 'WTA'

    def __init__(self, min_is_winner=False, **kwargs):
        self.min_is_winner = min_is_winner

    def f(self, yq, yh):
        winner_map = np.zeros(yh.shape)
        extreme = np.argmin(yh) if self.min_is_winner else np.argmax(yh)
        arg_ex = np.unravel_index(extreme, yh.shape)
        winner_map[arg_ex] = 1
        return winner_map - yh

    def f_derivative(self, yq, yh):
        return np.zeros(yh.shape)


def get_loss_function(name, **kwargs):
    if name == MseLoss.name:
        return MseLoss(**kwargs)
    elif name == WinnerTakeAllLoss.name:
        return WinnerTakeAllLoss(**kwargs)
    elif name == "WTA-min":
        return WinnerTakeAllLoss(min_is_winner=True, **kwargs)
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
