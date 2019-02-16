from NeuralNetworks.TicTacToe.framework import Player, Frame, Game
from numpy import array
import os
from keras.utils import to_categorical


class DenseNetworkPlayer(Player):

    MODEL_PATH = os.curdir + '/dense.h5'
    TYPE = 'dense'

    def get_positions(self, frame):
        frame = frame.matrix if self.character == Frame.X else Frame.flip(frame.matrix)
        processed_frame = array(Frame.categorize_inputs(array([frame]).reshape(1, 9)))
        model = DenseModel()
        output = model.model.predict(processed_frame)[0]
        output = self.get_max(output, frame)
        return Frame.output_linear_to_2D[output]

    def get_max(self, output, frame):
        while True:
            max_index = output.argmax()
            indices = Frame.output_linear_to_2D[max_index]
            if frame[indices[0]][indices[1]] is None:
                return max_index
            output[max_index] = -1


class DenseModel:

    def __init__(self):
        self.model = self.get_model()

    def train(self, epochs=250):
        data = Game.get_data()
        if not data['games']:
            return
        inputs, outputs = self.get_inputs_and_outputs(data)
        (X_train, Y_train), (X_test, Y_test) = self.preprocess(inputs, outputs)
        model = self.get_model()
        model.fit(X_train, Y_train, batch_size=128, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))
        model.save(DenseNetworkPlayer.MODEL_PATH)

    def get_model(self):
        if os.path.exists(DenseNetworkPlayer.MODEL_PATH):
            from keras.models import load_model
            model = load_model(DenseNetworkPlayer.MODEL_PATH)
        else:
            print("model not found!")
            from keras.models import Sequential
            from keras.layers import Dense
            model = Sequential()
            model.add(Dense(500, activation='relu', input_shape=(9,)))
            model.add(Dense(200, activation='relu'))
            model.add(Dense(9, activation='softmax'))
            model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
        return model

    def get_inputs_and_outputs(self, data):
        matches = data['games']
        inputs = []
        outputs = []
        for match in matches:
            for insert in match['match']:
                inputs.append(insert['frame'])
                outputs.append(insert['position'])

        return inputs, outputs

    def preprocess(self, inputs, outputs):

        inputs = array(Frame.categorize_inputs(array(inputs).reshape(len(inputs), 9)))

        outputs = to_categorical(self.linearize_outputs(outputs), 9)

        size = len(outputs)
        percent = int(size * 0.9)

        x_train = inputs[:percent]
        x_test = inputs[percent:]

        y_train = outputs[:percent]
        y_test = outputs[percent:]

        return (x_train, y_train), (x_test, y_test)

    def linearize_outputs(self, outputs):
        linear_outputs = []
        for output in outputs:
            linear_outputs.append(self.linearize_position(output[0], output[1]))
        return linear_outputs

    def linearize_position(self, row, column):
        count = 0
        for i in range(3):
            for j in range(3):
                if row == i and column == j:
                    return count
                count += 1


if __name__ == '__main__':
    DenseModel().train()
