from numpy import exp, random



class NeuralNetwork:
    def __init__(self, num_inputs, num_outputs, num_hidden_layer_neurons=None):
        self.layers = []
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        if num_hidden_layer_neurons is not None:
            for i in range(len(num_hidden_layer_neurons)):
                self.layers.append(Layer(num_hidden_layer_neurons[i], num_inputs))
                num_inputs = num_hidden_layer_neurons[i]
        self.layers.append(Layer(num_outputs, num_inputs))

    def train(self, inputs, outputs, iterations=10000):
        if len(inputs[0]) != self.num_inputs:
            raise Exception("Number of inputs for the neural network do not match")
        if len(outputs[0]) != self.num_outputs:
            raise Exception("Number of outputs for the nural network do not match")
        for iteration in range(iterations):
            for inputItr in range(len(inputs)):
                self.guess(inputs[inputItr])
                temp_outputs = list(outputs[inputItr])
                for layer in reversed(self.layers):
                    layer.adjust(temp_outputs)
                    temp_outputs = layer.get_outputs()

    def guess(self, inputs):
        for layer in self.layers:
            inputs = layer.fire(inputs)
        return inputs

    @staticmethod
    def activation_function(number):
        return 1 / (1 + exp(-number))

    @staticmethod
    def activation_derivative(number):
        return number*(1-number)


class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(num_inputs))

    def fire(self, inputs):
        return [neuron.fire(inputs) for neuron in self.neurons]

    def adjust(self, outputs):
        if len(outputs) != len(self.neurons):
            raise Exception("The number of outputs to adjust do not match with number of neurons.")
        for i in range(len(outputs)):
            self.neurons[i].adjust(outputs[i])

    def get_outputs(self):
        return [neuron.output for neuron in self.neurons]


class Neuron:
    def __init__(self, num_inputs, bias=0):
        self.weights = list(2 * random.random(num_inputs) - 1)
        self.bias = bias
        self.output = 0
        self.inputs = None

    def fire(self, inputs):
        if len(inputs) != len(self.weights):
            raise Exception("Number of inputs and number of weights are not same!")
        self.inputs = inputs
        result = 0
        for i in range(len(inputs)):
            result += inputs[i] * self.weights[i]
        result += self.bias
        result = NeuralNetwork.activation_function(result)
        self.output = result
        return result

    def adjust(self, output):
        multiplier = (output - self.output) * NeuralNetwork.activation_derivative(self.output)
        for i in range(len(self.weights)):
            self.weights[i] += self.inputs[i] * multiplier



def extend_lists(list1, list2):
    list3 = list(list1)
    list3.extend(list2)
    return list3


def main():
    days = [
        [1, 0, 0, 0, 0, 0, 0],  # S
        [0, 1, 0, 0, 0, 0, 0],  # M
        [0, 0, 1, 0, 0, 0, 0],  # T
        [0, 0, 0, 1, 0, 0, 0],  # W
        [0, 0, 0, 0, 1, 0, 0],  # T
        [0, 0, 0, 0, 0, 1, 0],  # F
        [0, 0, 0, 0, 0, 0, 1]  # S
    ]

    extra_inputs = [
        [1, 0],  # Today
        [0, 1]  # Tomorrow
    ]

    inputs = [
        # Today = ?
        extend_lists(days[0], extra_inputs[0]),
        extend_lists(days[1], extra_inputs[0]),
        extend_lists(days[2], extra_inputs[0]),
        extend_lists(days[3], extra_inputs[0]),
        extend_lists(days[4], extra_inputs[0]),
        extend_lists(days[5], extra_inputs[0]),
        extend_lists(days[6], extra_inputs[0]),
        # Tomorrow = ?
        extend_lists(days[0], extra_inputs[1]),
        extend_lists(days[1], extra_inputs[1]),
        extend_lists(days[2], extra_inputs[1]),
        extend_lists(days[3], extra_inputs[1]),
        extend_lists(days[4], extra_inputs[1]),
        extend_lists(days[5], extra_inputs[1]),
        extend_lists(days[6], extra_inputs[1])
    ]

    outputs = [
        # Today = ?
        days[1], days[2], days[3], days[4], days[5], days[6], days[0],
        # Tomorrow = ?
        days[2], days[3], days[4], days[5], days[6], days[0], days[1]
    ]

    rnn = NeuralNetwork(9, 7, [20])

    print("training began")
    rnn.train(inputs, outputs, 100)

    print(rnn.guess([0, 0, 0, 0, 1, 0, 0, 0, 1])) # Thu



main()
