import numpy as np
import random


class GdNpNeuron:

    def __init__(self, xs, ys):
        random.seed(6)
        self.xs = xs
        self.ys = ys
        self.adjusted_xs = self.get_adjusted_x(xs)
        self.weights = self.generate_random_coefficients(xs.shape[0])

    def equation(self):
        return self.adjusted_xs.T.dot(self.weights)

    def error(self, guess):
        return ((self.ys - guess) ** 2).sum()

    def error_diff(self, guess):
        weight_adjustment = self.adjusted_xs.dot(self.ys - guess)
        return weight_adjustment

    def learn(self, epochs, lr):
        print(f'X:\n{self.adjusted_xs}\nY:\n{self.ys}')
        for epoch in range(epochs):
            fx = self.equation()
            self.weights += lr * self.error_diff(fx)
            print(f'\nGuess:\n{fx}\nWeights:\n{self.weights}\nError = {self.error(fx)}')

    def test(self, x):
        return self.get_adjusted_x(x).T.dot(self.weights)

    @staticmethod
    def get_adjusted_x(x):
        ones = np.ones((1, x.shape[1]))
        adjusted_xs = np.append(ones, x, axis=0)
        return adjusted_xs

    @staticmethod
    def generate_random_coefficients(num_inputs):
        return np.random.uniform(-10, 10, (num_inputs + 1, 1))

    @staticmethod
    def generate_random_data(coefficients, data_size):
        # Get randomized Xs
        x = np.random.uniform(-10, 10, (len(coefficients) - 1, data_size))
        adjusted_x = GdNpNeuron.get_adjusted_x(x)

        # Get calculated Ys
        y = adjusted_x.T.dot(coefficients)

        return x, y


def main():
    weights = GdNpNeuron.generate_random_coefficients(2)
    x, y = GdNpNeuron.generate_random_data(weights, 7)
    test_x, test_y = GdNpNeuron.generate_random_data(weights, 3)

    print(f'\nWeights = {weights}')

    neuron = GdNpNeuron(x, y)
    neuron.learn(epochs=1000, lr=0.001)

    guesses = neuron.test(test_x)

    print(f'\nTest: Real = {test_y}\nGuess = {guesses}')


main()

