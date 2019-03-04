from NeuralNetworks.Snake.framework.match import Game
from NeuralNetworks.Snake.players.human import HumanPlayer


def main():
    player = HumanPlayer('human')
    game = Game(player)
    game.play(10)


if __name__ == '__main__':
    main()
