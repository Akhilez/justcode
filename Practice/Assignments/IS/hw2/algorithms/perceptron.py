import random

from data_manager import DataManager


class Perceptron:

    def __init__(self, xs, ys):
        self.xs = self.get_adjusted_x(xs)
        self.ys = ys
        self.weights = self.generate_random_coefficients(len(self.xs[0]))

    def learn(self, epochs, lr, x_test=None, y_test=None, grapher=None, test_grapher=None):
        for epoch in range(epochs):
            for i in range(len(self.xs)):
                fx = 0
                for j in range(len(self.weights)):
                    fx += self.xs[i][j] * self.weights[j]
                err = lr * (self.ys[i] - fx)
                for j in range(len(self.weights)):
                    self.weights[j] += err * self.xs[i][j]

            print(f'\nWeights:{self.weights}')

            if epoch == 0 or (epoch + 1) % 10 == 0:

                y_train_predicted = self.test(self.xs, need_bias_vector=False)
                hit_rate_train = DataManager.get_hit_rate(y_train_predicted, self.ys)

                print(f'Error = {1 - hit_rate_train}')

                if x_test:
                    y_test_predicted = self.test(x_test)
                    hit_rate_test = DataManager.get_hit_rate(y_test_predicted, y_test)
                    print(f'Test Error = {1 - hit_rate_test}')
                    if test_grapher:
                        test_grapher.record(epoch, 1 - hit_rate_test)

                if grapher:
                    grapher.record(epoch, 1 - hit_rate_train)

    def test(self, xs, need_bias_vector=True):
        xs = self.get_adjusted_x(xs) if need_bias_vector else xs
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
        x = list(x)
        for i in range(len(x)):
            one = [1]
            one.extend(x[i])
            x[i] = one
        return x

    @staticmethod
    def generate_random_coefficients(num_inputs):
        random.seed(0)
        return [random.uniform(-1, 1) for i in range(num_inputs)]
