from NeuralNetworks.Snake.framework.frame import Frame, Snake
from NeuralNetworks.Snake.framework.ui import UI


class Match:

    def __init__(self, player, ui):
        self.player = player
        self.ui = ui

    def start(self, frame, snake):
        frame.__init__()
        snake.__init__(frame)

        while not snake.is_dead():
            direction = self.player.get_direction(snake, self.ui)
            snake.move(direction)
            self.ui.update(snake, frame)

        else:
            # Game over.
            return


class Game:

    def __init__(self, player):
        self.frame = Frame()
        self.snake = Snake(self.frame)
        self.ui = UI(self.frame)
        self.player = player
        self.matches = []

    def play(self, count=1):
        for i in range(count):
            match = Match(self.player, self.ui)
            match.start(self.frame, self.snake)


