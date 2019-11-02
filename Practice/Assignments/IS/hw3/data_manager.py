import numpy as np


class DataManager:
    seed = 1

    def __init__(self, data_path=None, x_path='data/MNISTnumImages5000.txt', y_path='data/MNISTnumLabels5000.txt',
                 x_train_path='data/MNISTxTrain.txt', y_train_path='data/MNISTyTrain.txt',
                 x_test_path='data/MNISTxTest.txt', y_test_path='data/MNISTyTest.txt'):
        self._x_path = x_path
        self._y_path = y_path
        self._data_path = data_path
        self._x_train_path = x_train_path
        self._x_test_path = x_test_path
        self._y_train_path = y_train_path
        self._y_test_path = y_test_path

        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def split(self, randomized=True, percentage=0.8):
        if self.x is None:
            raise Exception("Must load the data by calling obj.load() before splitting.")
        n_rows, n_cols = self.x.shape
        split_index = int(percentage * n_rows)
        indices = self.get_randomized_indices(0, n_rows) if randomized else [i for i in range(n_rows)]
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        self.x_train = np.array([self.x[i] for i in train_indices])
        self.y_train = np.array([self.y[i] for i in train_indices])
        self.x_test = np.array([self.x[i] for i in test_indices])
        self.y_test = np.array([self.y[i] for i in test_indices])

        return self.x_train, self.y_train, self.x_test, self.y_test

    def load(self, split=False):
        if self._data_path:
            data = np.loadtxt(self._data_path)
            self.x = data[:, :-1]
            self.y = data[:, -1]
        elif split:
            self.x_train = np.loadtxt(self._x_train_path)
            self.y_train = np.loadtxt(self._y_train_path)
            self.x_test = np.loadtxt(self._x_test_path)
            self.y_test = np.loadtxt(self._y_test_path)
        else:
            self.x = np.loadtxt(self._x_path)
            self.y = np.loadtxt(self._y_path)

    @staticmethod
    def save(data, file_path):
        np.savetxt(file_path, data)

    @staticmethod
    def get_randomized_indices(min_, max_):
        indices = [i for i in range(min_, max_)]
        import random
        random.seed(DataManager.seed)
        DataManager.seed += 1
        random.shuffle(indices)
        return indices

    @staticmethod
    def _load_and_save_split_data():
        dm = DataManager()
        dm.load()
        x_train, y_train, x_test, y_test = dm.split()
        dm.save(x_train, 'data/MNISTxTrain.txt')
        dm.save(y_train, 'data/MNISTyTrain.txt')
        dm.save(x_test, 'data/MNISTxTest.txt')
        dm.save(y_test, 'data/MNISTyTest.txt')

