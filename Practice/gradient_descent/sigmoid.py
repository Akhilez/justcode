import math


class SigmoidGD:

    def __init__(self, w=0.1):
        self.w = w
        self.x = None
        self.y = None

    def load_data(self, xs, ys):
        self.x = xs
        self.y = ys

    def sigmoid(self, x):
        return 1 / (1 + (math.e ** (-1 * self.w * x)))

    def get_loss(self, y, guess):
        return (y - guess) ** 2

    def error_diff_w(self, guess, y, x):
        return (y - guess) * guess * (1 - guess) * x

    def train(self, epochs=400, lr=0.1):
        for epoch in range(epochs):
            for i in range(len(self.x)):
                print(f'\nWeight = {self.w}')
                print(f'x = {self.x[i]}, y = {self.y[i]}')

                guess = self.sigmoid(self.x[i])
                print(f'Guess = {guess}')

                error = self.get_loss(self.y[i], guess)
                print(f'Error = {error}')

                self.w += lr * self.error_diff_w(guess, self.y[i], self.x[i])


def main():
    neuron = SigmoidGD()
    # For equation => y = 1 / (1 + e^-(2.4x))
    neuron.load_data(
        [-3, -1, -0.5, -0.2, 0.2, 0.5, 0.8, 1, 1.2, 1.5, 2],
        [0, 0.08, 0.23, 0.38, 0.62, 0.77, 0.88, 0.92, 0.95, 0.97, 1]
    )  # This is sigmoid data
    neuron.train()
    print(f'for 0.5, fx = {neuron.sigmoid(0.5)}, y = 0.77')


main()
