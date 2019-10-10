import numpy as np
import random

from algorithms.knn import KNearestNeighbours


class Perceptron:

    def __init__(self, xs, ys):
        random.seed(6)
        self.xs = np.array(xs).T
        self.ys = np.array(ys).reshape((len(ys), 1))
        self.adjusted_xs = self.get_adjusted_x(self.xs)
        self.weights = self.generate_random_coefficients(self.xs.shape[0])

    def equation(self):
        return self.adjusted_xs.T.dot(self.weights)

    def equation_i(self, data_point):
        return data_point.dot(self.weights)

    def error(self, guess):
        return ((self.ys - guess) ** 2).sum()

    def error_diff(self, guess):
        weight_adjustment = self.adjusted_xs.dot(self.ys - guess)
        return weight_adjustment

    def learn(self, epochs, lr, grapher=None):

        # print(f'X:\n{self.adjusted_xs}\nY:\n{self.ys}')
        for epoch in range(epochs):
            xs = self.adjusted_xs.T
            guesses = []
            for i in range(len(xs)):
                fx = self.equation_i(xs[i])
                guesses.append(fx)
                for j in range(len(self.weights)):
                    err = self.ys[i][0] - fx
                    self.weights[j] += lr * err * xs[i][j]
            error = self.error(np.array(guesses))
            print(f'\nGuess:\nfx\nWeights:\n{self.weights}\nError = {error}')

            if grapher and epoch % 10 == 0:
                grapher.record(epoch, error)

    def test(self, x):
        return self.get_adjusted_x(np.array(x).T).T.dot(self.weights)

    @staticmethod
    def get_adjusted_x(x):
        x = np.array(x)
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
        adjusted_x = Perceptron.get_adjusted_x(x)

        # Get calculated Ys
        y = adjusted_x.T.dot(coefficients)

        return x, y


def main():
    weights = Perceptron.generate_random_coefficients(2)
    x, y = Perceptron.generate_random_data(weights, 7)
    test_x, test_y = Perceptron.generate_random_data(weights, 3)

    print(f'\nWeights = {weights}')

    neuron = Perceptron(x, y)
    neuron.learn(epochs=1000, lr=0.001)

    guesses = neuron.test(test_x)

    print(f'\nTest: Real = {test_y}\nGuess = {guesses}')
