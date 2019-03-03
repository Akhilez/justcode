import random


class Frame:

    LEFT = (-1, 0)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    UP = (0, -1)
    DIRECTIONS = (LEFT, DOWN, RIGHT, UP)

    def __init__(self, width=25, height=25):
        self.food_blocks = []
        self.canvas = []
        self.width = width
        self.height = height
        self.direction = self.DIRECTIONS[random.randint(0, 3)]

    def get_random_block(self):
        return (
            random.randint(0, self.width - 1),
            random.randint(0, self.height - 1)
        )

    def add_food(self, snake):
        self.food_blocks.append(self.get_next_food_block(snake))

    def get_next_food_block(self, snake):
        random_block = self.get_random_block()
        while random_block in snake.body:
            random_block = self.get_random_block()


class Snake:

    def __init__(self, frame):
        self.body = []
        self.frame = frame

    def move(self):
        self.enqueue()
        self.dequeue()

    def dequeue(self):
        if len(self.body) > 3:
            self.body.pop()

    def eat(self):
        self.enqueue()

    def enqueue(self):
        if len(self.body) == 0:
            self.body.insert(0, self.frame.get_random_block())
        else:
            head = self.body[0]
            self.body.insert(0, (
                head[0] + self.frame.direction[0],
                head[1] + self.frame.direction[1]
            ))

    def is_food_available(self):
        return len(self.frame.food_blocks) == 0 or self.body[0] == self.frame.food_blocks[-1]

    def is_dead(self):
        return self.has_eated_itself() or self.is_out_of_window()

    def has_eated_itself(self):
        return len(self.body) > 0 and self.body[0] in self.body[1:]

    def is_out_of_window(self):
        return any((
            self.body[0][0] >= self.frame.width,
            self.body[0][0] < 0,
            self.body[0][1] >= self.frame.height,
            self.body[0][1] < 0
        ))
