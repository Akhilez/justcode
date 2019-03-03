from NeuralNetworks.Snake.framework.frame import Frame, Snake


class Match:

    def __init__(self, player_1, player_2):
        self.player_1 = player_1
        self.player_2 = player_2
        self.current_player = player_1

    def start(self):
        frame = Frame()
        snake = Snake(frame)

        while not snake.is_dead():
            direction = self.current_player.get_direction()
            snake.move(direction)
        else:
            # Game over.
            return
