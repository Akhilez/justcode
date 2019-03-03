import pygame

from NeuralNetworks.Snake.framework.frame import Frame


class UI:
    SNAKE_COLOR = (30, 78, 24)
    FOOD_COLOR = (165, 24, 38)
    BG_COLOR = (156, 156, 156)

    def __init__(self, frame, speed=10):
        pygame.init()
        self.scale = 10
        self.speed = speed
        self.frame = frame
        self.display = pygame.display.set_mode((
            self.to_position(self.frame.width),
            self.to_position(self.frame.height)
        ))
        pygame.display.set_caption('Snake')

    def to_position(self, coordinate):
        return coordinate * self.scale

    def get_direction(self, default):
        direction = default
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and direction != Frame.RIGHT:
                    direction = Frame.LEFT
                if event.key == pygame.K_RIGHT and direction != Frame.LEFT:
                    direction = Frame.RIGHT
                if event.key == pygame.K_UP and direction != Frame.DOWN:
                    direction = Frame.UP
                if event.key == pygame.K_DOWN and direction != Frame.UP:
                    direction = Frame.DOWN
        return direction
