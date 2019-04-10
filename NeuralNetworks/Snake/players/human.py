from NeuralNetworks.Snake.framework.frame import Frame
from NeuralNetworks.Snake.players import Player


class HumanPlayer(Player):

    def get_direction(self, snake, ui):
        return ui.get_direction(snake.frame.direction)
