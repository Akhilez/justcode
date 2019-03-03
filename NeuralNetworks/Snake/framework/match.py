from NeuralNetworks.Snake.framework.frame import Frame, Snake


class Match:

    def start(self):
        frame = Frame()
        snake = Snake(frame)

        while not snake.is_dead():
            if snake.is_food_available():
                snake.eat()
            else:
                snake.move()
        else:
            # Game over.
            return
