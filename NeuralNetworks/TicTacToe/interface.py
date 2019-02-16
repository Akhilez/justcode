from NeuralNetworks.TicTacToe.framework import Frame, HumanPlayer, RandomPlayer, logger, Game
from NeuralNetworks.TicTacToe.smart_players import DenseNetworkPlayer, DenseModel


class TicTacToe:

    def create_console_game(self):
        """
        Read player 1 details
        Read player 2 details
        :return:
        """
        print("\nPlayer 1:")
        player_1_type = self.read_player_type()
        player_1_name = input('Enter player 1 name: ')
        player_1_character = self.read_character()

        print("\nPlayer 2:")
        player_2_type = self.read_player_type()
        player_2_name = input('Enter player 2 name: ')

        self.create_game(player_1_name, player_1_type, player_2_name, player_2_type, player_1_character).start()

    def create_automated_game(self, type_1, type_2):
        self.create_game(type_1, type_1, type_2, type_2, Frame.X).start(1000)

    def read_character(self):
        while True:
            character = input('Enter the player\'s character (X or O): ').upper()
            if character == Frame.X or character == Frame.O:
                return character
            print(f'Please enter either {Frame.X} or {Frame.O}')

    def create_game(self, player1_name=None, player1_type=HumanPlayer.TYPE, player2_name=None,
                    player2_type=HumanPlayer.TYPE, player1_character=Frame.X):

        player2_character = Frame.X if player1_character == Frame.O else Frame.O

        player1 = self.get_player(player1_type, player1_name, player1_character)
        player2 = self.get_player(player2_type, player2_name, player2_character)

        return Game(player1, player2)

    def get_player(self, player_type, player_name, player_character):
        if player_type == HumanPlayer.TYPE:
            return HumanPlayer(player_name, player_character)
        if player_type == RandomPlayer.TYPE:
            return RandomPlayer(player_name, player_character)
        if player_type == DenseNetworkPlayer.TYPE:
            return DenseNetworkPlayer(player_name, player_character)

    def read_player_type(self):
        while True:
            character = input('\n1. Human\n2. Randon\n3. Dense\nEnter the player type: ')
            if character not in '123':
                print("Wrong input")
            else:
                return {'1': HumanPlayer.TYPE, '2': RandomPlayer.TYPE, '3': DenseNetworkPlayer.TYPE}[character]

    def keep_dense_learning(self):
        model = DenseModel()
        # player2 = HumanPlayer('H1', Frame.X)
        player1 = DenseNetworkPlayer('D1', Frame.X)
        # player2 = DenseNetworkPlayer('D2', Frame.O)
        player2 = RandomPlayer('R1', Frame.O)

        for i in range(10):
            for i in range(10):
                player_temp = player2
                player2 = player1
                player1 = player_temp

                game = Game(player1, player2)

                game.start(1)
                old_data = Game.get_data()
                if old_data and 'games' in old_data and old_data['games']:
                    old_data['games'].pop(-1)
                    game.matches.extend(old_data['games'])
                game.save_data()
                model.train(50)

            Game.clear_data()


if __name__ == '__main__':
    tic_tac_toe = TicTacToe()
    # tic_tac_toe.create_automated_game(RandomPlayer.TYPE, RandomPlayer.TYPE)

    tic_tac_toe.keep_dense_learning()
