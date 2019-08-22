import math


class Neuron:
    def __init__(self, w=0.1, b=-0.7):
        self.w = w
        self.b = b
        self.x = None
        self.y = None

    def load_data(self, xs, ys):
        self.x = xs
        self.y = ys

    def train(self, epochs=10):
        for each in range(epochs):
            for i in range(len(self.x)):
                print(f'Weight = {self.w}')
                fx = self.sigmoid(self.w * self.x[i])
                print(f"y = {self.y[i]}, fx = {fx}")
                error = self.get_loss(self.y[i], fx)
                print(f'Error = {error}')
                self.w = self.w + (0.1 * self.differentiated_sigmoid(self.w * self.x[i]))
                print()


    def sigmoid(self, a):
        return 1 / (1 + math.e ** -a)

    def differentiated_sigmoid(self, a):
        sigmoid_answer = self.sigmoid(a)
        return sigmoid_answer * (1 - sigmoid_answer)

    def get_loss(self, y, fx):
        return (y-fx)**2


def main():
    neuron = Neuron()
    neuron.load_data([-1, -0.2, 0.2, 0.8], [0.08, 0.38, 0.62, 0.88])
    neuron.train()
    print(f'for 0.5, fx = {neuron.sigmoid(neuron.w * 0.5)}, y = 0.77')


main()
