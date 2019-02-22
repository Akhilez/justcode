import re
from abc import ABCMeta, abstractmethod
import random
import logging
import copy
import json
from functools import reduce

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class Frame:
    X = 'X'
    O = 'O'

    win_lines = [
        [(0, 0), (1, 1), (2, 2)],  # [\]
        [(0, 2), (1, 1), (2, 0)],  # [/]
        [(0, 0), (1, 0), (2, 0)],  # [|  ]
        [(0, 1), (1, 1), (2, 1)],  # [ | ]
        [(0, 2), (1, 2), (2, 2)],  # [  |]
        [(0, 0), (0, 1), (0, 2)],  # [```]
        [(1, 0), (1, 1), (1, 2)],  # [---]
        [(2, 0), (2, 1), (2, 2)],  # [...]
    ]

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
        for win_line in Frame.win_lines:
            num1 = self.matrix[win_line[0][0]][win_line[0][1]]
            num2 = self.matrix[win_line[1][0]][win_line[1][1]]
            num3 = self.matrix[win_line[2][0]][win_line[2][1]]

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
            random_index = random.randint(0, len(positions) - 1)
            return positions[random_index]


class Game:
    num_matches = 0

    def __init__(self, player1, player2):
        """
        :param player1: Player 1 (human|random|dense)
        :param player2: Player 2 (human|random|dense)
        """
        self.player_1, self.player_2 = player1, player2
        self.matches = []

    def start(self, epocs=None):
        while epocs is None or epocs > 0:
            match = Match(self.player_1, self.player_2, Game.num_matches)
            match.start()
            Game.num_matches += 1
            self.print_scores()
            # if match.win_status != [0, 0, 1]:
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

    def filter_draw_matches(self):
        return [match for match in self.matches if match.win_status != [0, 0, 1]]

    def swap_players(self):
        temp = self.player_1
        self.player_1 = self.player_2
        self.player_2 = temp


class Match:

    def __init__(self, player_1, player_2, id=None):
        self.frame = Frame()
        self.current_player = player_1
        self.other_player = player_2
        self.win_status = None
        self.inserts = []
        self.id = id

    def start(self):
        print(f"Match ID: {self.id}")
        while True:
            self.frame.print_canvas()
            print(f'Current player = {self.current_player} ({self.current_player.character})')
            self.insert(self.current_player.get_positions(self.frame))
            winner = self.frame.check_winner(self.current_player, self.other_player)
            if winner is not None or self.frame.is_canvas_filled():
                self.frame.print_canvas()
                self.print_winner(winner)
                self.update_scores(winner)
                self.win_status = self.get_win_status(None if winner is None else winner.character)
                return
            self.switch_players()

    def insert(self, positions):
        self.inserts.append({
            'current': self.current_player.character,
            'position': [positions[0], positions[1]],
            'frame': copy.deepcopy(self.frame.matrix)
        })
        self.frame.insert(self.current_player, positions[0], positions[1])

    @staticmethod
    def update_scores(winner):
        if winner is not None:
            winner.score += 1

    def summary(self):
        successful_inserts = self.get_best_inserts()
        # successful_inserts = self.remove_current_character_attribute(successful_inserts)
        return {'inserts': successful_inserts, 'id': self.id}

    def get_best_inserts(self):
        """
        Criteria to decide which inserts were the best:
        - Remove opportunity given
        - Remove missed opportunities
        - Add winner's inserts
        """
        best_inserts = []
        winner = self.get_winner(self.win_status)
        for insert in self.inserts:
            frame = Frame.flip(insert['frame']) if insert['current'] == Frame.O else copy.deepcopy(insert['frame'])
            new_insert = copy.deepcopy(insert)
            new_insert['frame'] = frame
            if all([
                # insert['current'] == winner,
                not self.has_missed_opportunity(frame, insert['current']),
                not self.is_opportunity_given(frame, insert['current']),
            ]):
                new_insert['best'] = True
            else:
                new_insert['best'] = False
            best_inserts.append(new_insert)
        return best_inserts

    @staticmethod
    def get_win_status(winner):
        win_status_getter = {Frame.X: [1, 0, 0], Frame.O: [0, 1, 0], None: [0, 0, 1]}
        return win_status_getter[winner]

    @staticmethod
    def get_winner(win_status):
        winner_getter = {(1, 0, 0): Frame.X, (0, 1, 0): Frame.O, (0, 0, 1): None}
        return winner_getter[tuple(win_status)]

    @staticmethod
    def print_winner(winner):
        if winner is None:
            print('Draw!')
        else:
            print(f'{winner} won!')

    def switch_players(self):
        switcher = self.current_player
        self.current_player = self.other_player
        self.other_player = switcher

    @staticmethod
    def remove_current_character_attribute(inserts):
        for insert in inserts:
            del insert['current']
        return inserts

    @staticmethod
    def is_opportunity_given(frame, current_position):
        """
        O O     O O
            ->
        X       X X
        Steps:
        - If current == O, flip the frame.
        - Check if O's have an opportunity:
            - In each win-line:
                - If it contains two O's and 1 None's and X is not placed in None: return True
        """
        for win_line in Frame.win_lines:
            num_o = sum(frame[position[0]][position[1]] == Frame.O for position in win_line)
            none_pos = [position for position in win_line if frame[position[0]][position[1]] is None]
            if num_o == 2 and len(none_pos) == 1 and none_pos[0] != current_position:
                return True
        return False

    @staticmethod
    def has_missed_opportunity(frame, current_position):
        """
        X X      X X
             ->  X
        OO       OO
        Steps:
        - If current == 0, flip the frame.
        - Check if X has missed an opportunity to win, but missed it.
            - In each win_line:
                - If it contains 2 X's and 1 None, and X is not placed in None: return True
        """
        for win_line in Frame.win_lines:
            num_x = sum(frame[position[0]][position[1]] == Frame.X for position in win_line)
            none_pos = [position for position in win_line if frame[position[0]][position[1]] is None]
            if num_x == 2 and len(none_pos) == 1 and none_pos[0] != current_position:
                return True
        return False


class DataManager:

    def __init__(self, file_name='data.json', max_size=10):
        self.file_name = file_name
        self.max_size = max_size

    def write(self, matches):
        with open(self.file_name, 'w') as data:
            data.write(json.dumps({'matches': matches}))

    def enqueue(self, matches):
        old_matches = self.get()
        matches.extend(old_matches)
        self.write(matches[:self.max_size])

    def get(self):
        try:
            with open(self.file_name, 'r') as data:
                data_string = data.read()
                if len(data_string) > 0:
                    return json.loads(data_string)['matches']
        except:
            pass
        return []

    def clear(self):
        try:
            with open(self.file_name, 'w') as data:
                data.write('')
        except FileNotFoundError:
            return
