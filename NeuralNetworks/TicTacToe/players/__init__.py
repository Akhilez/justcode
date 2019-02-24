from abc import ABCMeta, abstractmethod


class Player(metaclass=ABCMeta):
    TYPE = 'default'

    def __init__(self, name, character=None):
        self.name = name
        self.total_score = 0
        self.total_games = 0
        self.score = 0
        self.character = self.get_character(character)

    @abstractmethod
    def get_positions(self, frame):
        pass

    def get_character(self, character):
        if character is None:
            while True:
                character = input('Enter player 1 character (X or O): ').upper()
                if character == Frame.X or character == Frame.O:
                    break
                print(f'Please enter either {Frame.X} or {Frame.O}')
        return character

    def __str__(self):
        return self.character

    def __eq__(self, other):
        return self.character == other.character
