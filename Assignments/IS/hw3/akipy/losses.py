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
        extreme = np.argmin(yh) if self.min_is_winner else np.argmax(yh)
        arg_ex = np.unravel_index(extreme, yh.shape)
        return yh[arg_ex]

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
    def decay(self, time_i, **kwargs):
        pass


class ReverseExponentialDecay(RateDecay):
    def __init__(self, time_constant=5):
        self.time_constant = time_constant

    def decay(self, time_i, **kwargs):
        return math.exp(-1 * time_i / self.time_constant)

