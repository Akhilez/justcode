from NeuralNetworks.TicTacToe.framework import Player, Frame
from numpy import array
import json
import os


class DenseNetworkPlayer(Player):

    MODEL_PATH = os.curdir + '/dense.h5'
    TYPE = 'dense'
    output_categories = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (2, 0), 7: (2, 1),
                         8: (2, 2)}

    def get_positions(self, frame):
        frame = frame.matrix if self.character == Frame.X else Frame.flip(frame.matrix)
        processed_frame = array(self.categorize_inputs(array([frame]).reshape(1, 9)))
        model = self.get_model()
        output = model.predict(processed_frame)[0].argmax()
        return DenseNetworkPlayer.output_categories[output]

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

    def train(self, epochs=250):
        inputs, outputs = self.get_data()
        (X_train, Y_train), (X_test, Y_test) = self.preprocess(inputs, outputs)
        model = self.get_model()
        model.fit(X_train, Y_train, batch_size=128, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))
        model.save(DenseNetworkPlayer.MODEL_PATH)

    def get_data(self):
        data = open('data.json', 'r').read()
        matches = json.loads(data)['game']

        inputs = []
        outputs = []
        for match in matches:
            for insert in match['match']:
                inputs.append(insert['frame'])
                outputs.append(insert['position'])

        return inputs, outputs

    def preprocess(self, inputs, outputs):
        from keras.utils import to_categorical

        inputs = array(self.categorize_inputs(array(inputs).reshape(len(inputs), 9)))

        outputs = to_categorical(self.linearize_outputs(outputs))

        size = len(outputs)
        percent = int(size * 0.8)

        x_train = inputs[:percent]
        x_test = inputs[percent:]

        y_train = outputs[:percent]
        y_test = outputs[percent:]

        return (x_train, y_train), (x_test, y_test)

    def categorize_inputs(self, my_list):
        categories = {None: 0.0, 'X': 0.5, 'O': 1.0}
        all_list = []
        for frame in my_list:
            category_list = []
            for position in frame:
                category_list.append(categories[position])
            all_list.append(category_list)
        return all_list

    def linearize_position(self, row, column):
        count = 0
        for i in range(3):
            for j in range(3):
                if row == i and column == j:
                    return count
                count += 1

    def linearize_outputs(self, outputs):
        linear_outputs = []
        for output in outputs:
            linear_outputs.append(self.linearize_position(output[0], output[1]))
        return linear_outputs


if __name__ == '__main__':
    player = DenseNetworkPlayer('trainer')
    player.train()
