import math


class Neuron:

    # Equation of the line => y = wx + b
    # Equation of the error => f(x, y) = (y - (wx + b))**2
    # Variables => w, b
    # To learn w:
    #   Differentiate error function f(x, y) with respect to w => 2 * (y - (wx + b)) * x
    #   Add this value to the current w
    #   w = w + (y - (wx + b)) * x
    # To learn b:
    #   Differentiate error function f(x, y) wrt b => 2 * (y - (wx + b))
    #   Add this value to b
    #   b = b + (y - (wx + b))

    def __init__(self, w=0.1, b=-0.7):
        self.w = w
        self.b = b
        self.x = None
        self.y = None

    def load_data(self, xs, ys):
        self.x = xs
        self.y = ys

    def equation(self, x):
        return x * self.w + self.b

    def error(self, guess, y):
        return (y - guess)**2

    def error_diff_w(self, guess, y, x):
        return (y - guess) * x

    def error_diff_b(self, guess, y):
        return y - guess

    def learn(self, epochs=10, lr=0.1):
        for epoch in range(epochs):
            for i in range(len(self.x)):
                print(f'\nWeight = {self.w}, Bias = {self.b}')
                print(f'x = {self.x[i]}, y = {self.y[i]}')

                guess = self.equation(self.x[i])
                print(f'Guess = {guess}')

                error = self.error(guess, self.y[i])
                print(f'Error = {error}')

                self.w += lr * self.error_diff_w(guess, self.y[i], self.x[i])
                self.b += lr * self.error_diff_b(guess, self.y[i])

    def sigmoid(self, a):
        return 1 / (1 + math.e ** -a)

    def differentiated_sigmoid(self, a):
        sigmoid_answer = self.sigmoid(a)
        return sigmoid_answer * (1 - sigmoid_answer)

    def get_loss(self, y, fx):
        return (y - fx) ** 2


def main():
    neuron = Neuron()
    # neuron.load_data([-1, -0.2, 0.2, 0.8], [0.08, 0.38, 0.62, 0.88])  # This is sigmoid data
    # neuron.train()
    # print(f'for 0.5, fx = {neuron.sigmoid(neuron.w * 0.5)}, y = 0.77')

    # Equation of the current line => y = 2x - 5
    neuron.load_data([-1, -0.7, 0.2, 0.8, 3, 4], [-7, -6.4, -4.6, -3.4, 1, 3])  # This is a line data
    neuron.learn()
    test = neuron.equation(2)  # Ans = -1

    print(f'\nTest: for 2, guess = {test}, real = -1')


main()
