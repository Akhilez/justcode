from NeuralNetworks.TicTacToe.framework import Player, Frame, Game, DataManager
import numpy as np
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class DenseNetworkPlayer(Player):

    TYPE = 'dense'

    def __init__(self, name, character=None):
        super().__init__(name, character)
        self.model = DenseModel(self.name)

    def get_positions(self, frame):
        frame = frame.matrix if self.character == Frame.X else Frame.flip(frame.matrix)
        processed_frame = np.array([self.model.categorize_inputs(frame)]).reshape(1, 27)
        output = self.model.model.predict(processed_frame)[0]
        output = self.get_max(output, frame)
        return [int(output // 3), int(output % 3)]

    @staticmethod
    def get_max(output, frame):
        while True:
            max_index = output.argmax()
            indices = [max_index // 3, max_index % 3]
            if frame[indices[0]][indices[1]] is None:
                return max_index
            output[max_index] = -1


class DenseModel:

    def __init__(self, model_name):
        self.name = model_name
        self.model_path = f'{os.curdir}/{self.name}.h5'
        self.model = self.get_model()

    def train(self, epochs=250, data_manager=None):
        data_manager = data_manager if data_manager else DataManager()
        matches = data_manager.get()
        x_train, x_test, y_train, y_test = self.get_inputs_and_outputs(matches)
        model = self.get_model()
        model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
        model.save(self.model_path)

    def get_model(self):
        if os.path.exists(self.model_path):
            from keras.models import load_model
            model = load_model(self.model_path)
        else:
            print("model not found!")
            from keras.models import Sequential
            from keras.layers import Dense
            model = Sequential()
            model.add(Dense(3000, activation='relu', input_shape=(27,)))
            model.add(Dense(1000, activation='relu'))
            model.add(Dense(9, activation='softmax'))
            model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
        return model

    def get_inputs_and_outputs(self, matches):
        inputs = []
        outputs = []
        for match in matches:
            for insert in match['inserts']:
                inputs.append(self.categorize_inputs(insert['frame']))
                outputs.append(insert['position'])
        outputs = np.array(self.categorize_outputs(outputs))
        inputs = np.array(inputs).reshape(len(inputs), 27)

        return train_test_split(inputs, outputs, test_size=0.2)

    def categorize_inputs(self, my_list):
        categories = {None: [0, 0, 1], 'X': [1, 0, 0], 'O': [0, 1, 0]}
        all_list = []
        for frame in my_list:
            category_list = []
            for position in frame:
                category_list.append(categories[position])
            all_list.append(category_list)
        return all_list

    def categorize_outputs(self, my_list):
        cat_list = []
        for lst in my_list:
            cat_list.append(lst[0] * 3 + lst[1])
        return to_categorical(cat_list, 9)


if __name__ == '__main__':
    DenseModel('testing').train()
