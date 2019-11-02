from abc import ABC, abstractmethod


class LossFunction(ABC):
    pass


class MseLoss(LossFunction):
    name = 'MSE'


def get_loss_function(name, **kwargs):
    if name == 'MSE':
        return MseLoss(**kwargs)
    else:
        raise Exception(f"Loss function {name} is not found.")
