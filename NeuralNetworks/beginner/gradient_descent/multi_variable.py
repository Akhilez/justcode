import random


class MultiVariableGD:
    """
    Equation: y = (w0 * x0) + (w1 * x1) + (w2 * x2) + (w3 * x3) + (w4 * x4)
    Parameters: w0, w1, w2, w3, w4
    Error: (y - f(x1, x2, x3, x4)) ** 2
        (y - ((w0 * 1) + (w1 * x1) + (w2 * x2) + (w3 * x3) + (w4 * x4))) ** 2
    Error differentiation wrt wi: 2 * (y - f(x)) * -1 * xi
    Note: x0 is ALWAYS 1
    """

    def __init__(self, x, y):
        random.seed(6)
        self.x = x
        self.y = y
        self.adjusted_x = self.get_adjusted_x()
        self.weights = self.initialize_weights()

    def get_adjusted_x(self):
        adjusted_x = []
        for i in self.x:
            x = [1]
            x.extend(i)
            adjusted_x.append(x)
        return adjusted_x

    def initialize_weights(self):
        weights = []
        for i in range(len(self.adjusted_x[0])):
            weights.append(random.random())
        return weights

    def equation(self, x):
        result = 0
        for i in range(len(x)):
            result += x[i] * self.weights[i]
        return result

    def error(self, guess, y):
        return (y - guess) ** 2

    def error_diff(self, guess, y, x):
        return (y - guess) * x

    def learn(self, epochs=10, lr=0.01):
        for epoch in range(epochs):
            for i in range(len(self.adjusted_x)):
                print(f'\nWeights = {self.weights}')
                print(f'x = {self.adjusted_x[i]}, y = {self.y[i]}')

                guess = self.equation(self.adjusted_x[i])
                print(f'Guess = {guess}')

                error = self.error(guess, self.y[i])
                print(f'Error = {error}')

                for wi in range(len(self.weights)):
                    self.weights[wi] += lr * self.error_diff(guess, self.y[i], self.adjusted_x[i][wi])

    @staticmethod
    def generate_random_coefficients(num_inputs):
        # Get randomized coefficients
        coefficients = []
        for i in range(num_inputs + 1):
            coefficients.append(random.uniform(-10.0, 10.0))
        return coefficients

    @staticmethod
    def generate_random_data(coefficients, data_size):
        # Get randomized Xs
        x = []
        for i in range(data_size):
            xi = []
            for j in range(len(coefficients) - 1):
                xi.append(random.uniform(-10.0, 10.0))
            x.append(xi)

        # Get calculated Ys
        y = []
        for i in range(data_size):
            result = coefficients[0]
            for xi in range(len(x[i])):
                result += x[i][xi] * coefficients[xi + 1]
            y.append(result)

        return x, y

    def test(self, x):
        tests = []
        for i in range(len(x)):
            adjusted_x = [1]
            adjusted_x.extend(x[i])
            tests.append(self.equation(adjusted_x))
        return tests


def main():
    weights = MultiVariableGD.generate_random_coefficients(2)
    x, y = MultiVariableGD.generate_random_data(weights, 7)
    test_x, test_y = MultiVariableGD.generate_random_data(weights, 3)

    print(f'\nWeights = {weights}')
    for i in range(len(x)):
        print(f'X = {x[i]}, y = {y[i]}')

    neuron = MultiVariableGD(x, y)
    neuron.learn(epochs=100, lr=0.01)

    guesses = neuron.test(test_x)

    for i in range(len(test_x)):
        print(f'\nTest: Real = {test_y[i]}, Guess = {guesses[i]}')


main()
