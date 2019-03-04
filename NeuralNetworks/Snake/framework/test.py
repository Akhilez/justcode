import pygame
import random


class Snake:

    LEFT = (-1, 0)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    UP = (0, -1)
    DIRECTIONS = (LEFT, DOWN, RIGHT, UP)
    SNAKE_COLOR = (30, 78, 24)
    FOOD_COLOR = (165, 24, 38)
    BG_COLOR = (156, 156, 156)

    def __init__(self, sped=10, num_matches=10):
        pygame.init()
        self.scale = 10
        self.width = 30
        self.height = 30
        self.speed = 10
        self.display = pygame.display.set_mode((self.to_position(self.width), self.to_position(self.height)))
        pygame.display.set_caption('Snake')

        self.blocked = False
        self.direction = self.DIRECTIONS[random.randint(0, 3)]
        self.snake = []
        self.food_blocks = []

    def start(self):
        clock = pygame.time.Clock()
        while not self.blocked:
            self.listen_change_in_direction()

            self.move_in_direction()

            self.sync_display()

            if self.quit():
                self.blocked = True

            pygame.display.update()
            clock.tick(self.speed)

        pygame.quit()
        quit()

    def listen_change_in_direction(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction != self.RIGHT:
                    self.direction = self.LEFT
                if event.key == pygame.K_RIGHT and self.direction != self.LEFT:
                    self.direction = self.RIGHT
                if event.key == pygame.K_UP and self.direction != self.DOWN:
                    self.direction = self.UP
                if event.key == pygame.K_DOWN and self.direction != self.UP:
                    self.direction = self.DOWN

    def get_random_block(self):
        return (
            random.randint(0, self.width - 1),
            random.randint(0, self.height - 1)
        )

    def to_position(self, coordinate):
        return coordinate * self.scale

    def move_in_direction(self):
        """
        if snake is empty, create its head at random position.
        Enqueue snake in direction.
        If snake head == body, then Game Over.
        if snake head is out of bounds, Game Over.
        if sname head == food, increase snake size
        else deque the snake.
        """

        # Enqueuing
        if len(self.snake) == 0:
            self.snake.insert(0, self.get_random_block())
        else:
            head = self.snake[0]
            self.snake.insert(0, (
                head[0] + self.direction[0],
                head[1] + self.direction[1]
            ))

        # Eats itself
        if len(self.snake) > 0 and self.snake[0] in self.snake[1:]:
            self.blocked = True

        # Goes out of window
        if self.snake[0][0] >= self.width or self.snake[0][0] < 0 or self.snake[0][1] >= self.height or self.snake[0][1] < 0:
            self.blocked = True

        # Ate food!
        if len(self.food_blocks) == 0 or self.snake[0] == self.food_blocks[-1]:
            random_block = self.get_random_block()
            while random_block in self.snake:
                random_block = self.get_random_block()
            self.food_blocks.append(random_block)
        elif len(self.snake) > 3:
            self.snake.pop()

    def sync_display(self):
        self.display.fill(self.BG_COLOR)

        for snake_block in self.snake:
            pygame.draw.rect(self.display, self.SNAKE_COLOR, (self.to_position(snake_block[0]), self.to_position(snake_block[1]), self.scale, self.scale), 0)

        if len(self.food_blocks) > 0:
            pygame.draw.rect(self.display, self.FOOD_COLOR,
                             (self.to_position(self.food_blocks[-1][0]), self.to_position(self.food_blocks[-1][1]), self.scale, self.scale), 0)

    def quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False


def main():
    Snake().start()


if __name__ == '__main__':
    main()
