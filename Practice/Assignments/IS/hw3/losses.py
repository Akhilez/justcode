from abc import ABC, abstractmethod


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
        return -1 * (yq - yh)


def get_loss_function(name, **kwargs):
    if name == MseLoss.name:
        return MseLoss(**kwargs)
    else:
        raise Exception(f"Loss function {name} is not found.")
