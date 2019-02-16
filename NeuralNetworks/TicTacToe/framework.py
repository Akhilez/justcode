import re
from abc import ABCMeta, abstractmethod
import random
import logging
import copy
import json

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class Frame:

    X = 'X'
    O = 'O'
    output_linear_to_2D = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (2, 0), 7: (2, 1),
                         8: (2, 2)}

    def __init__(self):
        self.matrix = self.generate_empty_canvas()

    def insert(self, player, row, column):
        self.matrix[row][column] = player.character

    def print_canvas(self):
        output = '\n\t0\t1\t2\n'
        for i in range(3):
            output += f'{i}\t'
            for j in range(3):
                value = self.matrix[i][j]
                value = value if value is not None else ' '
                output += f'{value}\t'
            output += '\n'
        output += '\n'
        print(output)

    def check_winner(self, player1, player2):
        checks = [
            [(0, 0), (1, 1), (2, 2)],  # [\]
            [(0, 2), (1, 1), (2, 0)],  # [/]
            [(0, 0), (1, 0), (2, 0)],  # [|  ]
            [(0, 1), (1, 1), (2, 1)],  # [ | ]
            [(0, 2), (1, 2), (2, 2)],  # [  |]
            [(0, 0), (0, 1), (0, 2)],  # [```]
            [(1, 0), (1, 1), (1, 2)],  # [---]
            [(2, 0), (2, 1), (2, 2)],  # [...]
        ]
        for check in checks:
            num1 = self.matrix[check[0][0]][check[0][1]]
            num2 = self.matrix[check[1][0]][check[1][1]]
            num3 = self.matrix[check[2][0]][check[2][1]]

            if num1 is not None and num1 == num2 and num2 == num3:
                return player1 if player1.character == num1 else player2

    def is_canvas_filled(self):
        for row in self.matrix:
            for column in row:
                if column is None:
                    return False
        return True

    def generate_empty_canvas(self):
        return [
            [None, None, None],
            [None, None, None],
            [None, None, None]
        ]

    @staticmethod
    def categorize_inputs(my_list):
        categories = {None: 0.0, 'X': 0.5, 'O': 1.0}
        all_list = []
        for frame in my_list:
            category_list = []
            for position in frame:
                category_list.append(categories[position])
            all_list.append(category_list)
        return all_list

    @staticmethod
    def flip(matrix):
        flipped = copy.deepcopy(matrix)
        for i in range(3):
            for j in range(3):
                if matrix[i][j] is None:
                    continue
                if matrix[i][j] == Frame.X:
                    flipped[i][j] = Frame.O
                else:
                    flipped[i][j] = Frame.X
        return flipped

    @staticmethod
    def linear(matrix):
        linear_matrix = []
        for i in range(3):
            for j in range(3):
                linear_matrix.append(matrix[i][j])
        return linear_matrix

    @staticmethod
    def linearize_position(row, column):
        count = 0
        for i in range(3):
            for j in range(3):
                if row == i and column == j:
                    return count
                count += 1


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
        return self.name

    def __eq__(self, other):
        return self.character == other.character


class HumanPlayer(Player):

    TYPE = 'human'

    def get_positions(self, frame):
        while True:
            positions = input('Enter position in "x y" format: ').strip()
            inputs = re.match(r'([0-2])[\s,-]+([0-2])', positions)
            if inputs is not None:
                inputs = inputs.groups()
                inputs = int(inputs[0]), int(inputs[1])
                if frame.matrix[inputs[0]][inputs[1]] is not None:
                    print('That position already has a value.')
                    continue
                return inputs
            print("Wrong input.")


class RandomPlayer(Player):

    TYPE = 'random'

    def get_positions(self, frame):
        positions = []
        for i in range(3):
            for j in range(3):
                if frame.matrix[i][j] is None:
                    positions.append((i, j))
        if len(positions) > 0:
            random_index = random.randint(0, len(positions)-1)
            return positions[random_index]


class Game:

    def __init__(self, player1, player2):
        """
        :param player1: Player 1 (human|random|dense)
        :param player2: Player 2 (human|random|dense)
        """
        self.player_1, self.player_2 = player1, player2
        self.matches = []

    def start(self, epocs=None):
        while epocs is None or epocs > 0:
            match = Match(self.player_1, self.player_2)
            match.start()
            self.print_scores()
            if match.win_status != [0, 0, 1]:
                match_summary = match.summary()
                self.matches.append(match_summary)
            if epocs is None:
                if self.choose_to_replay():
                    continue
                else:
                    print("Closing the game. Bye!")
                    epocs = 0
            else:
                epocs -= 1

    def choose_to_replay(self):
        choice = input("Replay? (y/n):").lower()
        return choice == 'y'

    def print_scores(self):
        print(f"Scores:\n\t{self.player_1}: {self.player_1.score}")
        print(f"\t{self.player_2}: {self.player_2.score}")

    def save_data(self):
        with open('data.json', 'w') as data:
            data.write(json.dumps({'games': self.matches}))

    def filter_draw_matches(self):
        return [match for match in self.matches if match.win_status != [0, 0, 1]]

    @staticmethod
    def get_data():
        try:
            with open('data.json', 'r') as data:
                data_string = data.read()
                if len(data_string) > 0:
                    return json.loads(data_string)
        except FileNotFoundError:
            return

    @staticmethod
    def clear_data():
        try:
            with open('data.json', 'w') as data:
                data.write('')
        except FileNotFoundError:
            return


class Match:

    def __init__(self, player_1, player_2):
        self.frame = Frame()
        self.current_player = player_1
        self.other_player = player_2
        self.win_status = None
        self.inserts = []

    def start(self):
        while True:
            self.frame.print_canvas()
            print(f'Current player = {self.current_player} ({self.current_player.character})')
            self.insert(self.current_player.get_positions(self.frame))
            winner = self.frame.check_winner(self.current_player, self.other_player)
            if winner is not None or self.frame.is_canvas_filled():
                self.frame.print_canvas()
                self.print_winner(winner)
                self.update_scores(winner)
                self.win_status = self.get_win_status(winner)
                return
            self.switch_players()

    def insert(self, positions):
        frame = copy.deepcopy(self.frame.matrix) if self.current_player.character == Frame.X else Frame.flip(self.frame.matrix)
        self.inserts.append({
            'current': self.current_player.character,
            'position': [positions[0], positions[1]],
            'frame': frame
        })
        self.frame.insert(self.current_player, positions[0], positions[1])

    def update_scores(self, winner):
        if winner is not None:
            winner.score += 1

    def summary(self):
        successful_inserts = self.get_successful_inserts()
        successful_inserts = self.remove_current_character_attribute(successful_inserts)
        return {'match': successful_inserts}

    def get_successful_inserts(self):
        successful_inserts = []
        if self.win_status == [1, 0, 0]:
            drop_character = Frame.O
        else:
            drop_character = Frame.X
        for i in range(len(self.inserts)):
            if self.inserts[i]['current'] != drop_character:
                successful_inserts.append(self.inserts[i])
        return successful_inserts

    def get_win_status(self, winner):
        if winner is None:
            return [0, 0, 1]
        if winner.character == Frame.X:
            return [1, 0, 0]
        return [0, 1, 0]

    def print_winner(self, winner):
        if winner is None:
            print('Draw!')
        else:
            print(f'{winner} won!')

    def switch_players(self):
        switcher = self.current_player
        self.current_player = self.other_player
        self.other_player = switcher

    def remove_current_character_attribute(self, inserts):
        for insert in inserts:
            del insert['current']
        return inserts
