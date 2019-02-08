import re


class Frame:

    X = 'X'
    O = 'O'

    def __init__(self):
        self.matrix = self.generate_empty_canvas()

    def insert(self, player, row, column):
        self.matrix[row][column] = player.character

    def print_canvas(self):
        print('\n\t0\t1\t2')
        for i in range(3):
            print(i, end='\t')
            for j in range(3):
                value = self.matrix[i][j]
                value = value if value is not None else ' '
                print(value, end='\t')
            print()
        print()

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


class Player:
    def __init__(self, name, character=Frame.X):
        self.name = name
        self.total_score = 0
        self.total_games = 0
        self.character = character
        self.score = 0

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.character == other.character


class Game:

    def __init__(self):
        self.player_1, self.player_2 = self.get_players()
        self.matches = []

    def start(self):
        match = Match(self.player_1, self.player_2)
        match.start()
        self.print_scores()
        self.matches.append(match.summary())
        if self.choose_to_replay():
            self.start()
        else:
            print("Closing the game. Bye!")

    def choose_to_replay(self):
        choice = input("Replay? (y/n):").lower()
        return choice == 'y'

    def get_players(self):
        player1_name = input('Enter player 1 name: ')
        while True:
            player1_character = input('Enter player 1 character (X or O): ').upper()
            if player1_character == Frame.X or player1_character == Frame.O:
                break
            print(f'Please enter either {Frame.X} or {Frame.O}')
        player2_name = input('Enter player 2 name: ')
        player2_character = Frame.X if player1_character == Frame.O else Frame.O

        return (
            Player(player1_name, player1_character),
            Player(player2_name, player2_character)
        )

    def print_scores(self):
        print(f"Scores:\n\t{self.player_1}: {self.player_1.score}")
        print(f"\t{self.player_2}: {self.player_2.score}")


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
            self.insert(self.read_positions())
            winner = self.frame.check_winner(self.current_player, self.other_player)
            if winner is not None or self.frame.is_canvas_filled():
                self.frame.print_canvas()
                self.print_winner(winner)
                self.update_scores(winner)
                self.win_status = self.get_win_status(winner)
                return
            self.switch_players()

    def insert(self, positions):
        self.inserts.append((positions[0], positions[1], self.current_player.character))
        self.frame.insert(self.current_player, positions[0], positions[1])

    def update_scores(self, winner):
        if winner is not None:
            winner.score += 1

    def summary(self):
        return {
            'frame': self.frame.matrix,
            'inserts': self.inserts,
            'win_status': self.win_status
        }

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

    def read_positions(self):
        while True:
            positions = input('Enter position in "x y" format: ').strip()
            inputs = re.match(r'([0-2])[\s,-]+([0-2])', positions)
            if inputs is not None:
                inputs = inputs.groups()
                inputs = int(inputs[0]), int(inputs[1])
                if self.frame.matrix[inputs[0]][inputs[1]] is not None:
                    print('That position already has a value.')
                    continue
                return inputs
            print("Wrong input.")


if __name__ == '__main__':
    game = Game()
    game.start()
    print(game.matches)
