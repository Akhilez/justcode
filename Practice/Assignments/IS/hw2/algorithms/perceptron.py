import random


class Perceptron:

    def __init__(self, xs, ys):
        random.seed(6)
        self.xs = self.get_adjusted_x(xs)
        self.ys = ys
        self.weights = self.generate_random_coefficients(len(self.xs[0]))

    def learn(self, epochs, lr, grapher=None):
        for epoch in range(epochs):
            guesses = []
            error = 0
            for i in range(len(self.xs)):
                fx = 0
                for j in range(len(self.weights)):
                    fx += self.xs[i][j] * self.weights[j]
                guesses.append(fx)
                err = lr * (self.ys[i] - fx)
                error += (self.ys[i] - fx) ** 2
                for j in range(len(self.weights)):
                    self.weights[j] += err * self.xs[i][j]

            print(f'\nWeights:{self.weights}\nError = {error}')

            if grapher or epoch % 10 == 0:
                grapher.record(epoch, error)

    def test(self, xs):
        xs = self.get_adjusted_x(xs)
        ys = []
        for i in range(len(xs)):
            y = 0
            for j in range(len(xs[i])):
                y += xs[i][j] * self.weights[j]
            y = 1 if y > 0.5 else 0
            ys.append(y)
        return ys

    @staticmethod
    def get_adjusted_x(x):
        for i in range(len(x)):
            one = [1]
            one.extend(x[i])
            x[i] = one
        return x

    @staticmethod
    def generate_random_coefficients(num_inputs):
        return [random.uniform(-1, 1) for i in range(num_inputs)]
