from abc import ABCMeta, abstractmethod


class Player(metaclass=ABCMeta):
    TYPE = 'default'

    def __init__(self, name, pygame):
        self.name = name
        self.score = 0
        self.pygame = pygame

    @abstractmethod
    def get_direction(self, snake, ui):
        pass

    def __str__(self):
        return f'{self.name})'
