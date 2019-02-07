

class Frame:

    X = 'X'
    O = 'O'

    def __init__(self):
        self.matrix = self.generate_empty_canvas()
        self.player1 = Player()
        self.player2 = Player(first=False)

    def insert(self, player, row, column):
        self.matrix[row][column] = player.character

    def print_canvas(self):
        print(self.matrix[0])
        print(self.matrix[1])
        print(self.matrix[2])

    def check_winner(self):
        checks = [
            [(0, 0), (1, 1), (2, 2)],  # [\]
            [(0, 0), (1, 0), (2, 0)],  # [|  ]
            [(0, 1), (1, 1), (2, 1)],  # [ | ]
            [(0, 2), (1, 2), (2, 2)],  # [  |]
            [(0, 2), (1, 1), (2, 0)],  # [/]
            [(0, 0), (0, 1), (0, 2)],  # [```]
            [(2, 0), (2, 1), (2, 2)],  # [...]
            [(1, 0), (1, 1), (1, 2)],  # [---]
        ]
        for check in checks:
            num1 = self.matrix[check[0][0]][check[0][1]]
            num1 = self.matrix[check[0][0]][check[0][1]]
            num1 = self.matrix[check[0][0]][check[0][1]]

    def generate_empty_canvas(self):
        return [
            [None, None, None],
            [None, None, None],
            [None, None, None]
        ]

class Player:
    def __init__(self, first=True):
        self.total_score = 0
        self.total_games = 0
        self.character = Frame.X if first else Frame.O
