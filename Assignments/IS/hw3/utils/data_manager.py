import numpy as np


class DataManager:

    def __init__(self, data_path=None, x_path=None, y_path=None, x_train_path=None, y_train_path=None, x_test_path=None,
                 y_test_path=None):
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

    def load(self, split=False, one_hot=False):
        if self._data_path:
            data = np.loadtxt(self._data_path)
            self.x = data[:, :-1]
            self.y = data[:, -1]
            if one_hot:
                self.y = self.get_one_hot(self.y)
            if split:
                self.split()
        elif split:
            self.x_train = np.load(self._x_train_path)
            self.y_train = np.load(self._y_train_path)
            self.x_test = np.load(self._x_test_path)
            self.y_test = np.load(self._y_test_path)
            if one_hot:
                self.y_train = self.get_one_hot(self.y_train)
                self.y_test = self.get_one_hot(self.y_test)
        else:
            self.x = np.loadtxt(self._x_path)
            self.y = np.loadtxt(self._y_path)
            if one_hot:
                self.y = self.get_one_hot(self.y)

    @staticmethod
    def get_one_hot(y):
        unique_y = np.unique(y)
        unique_y.sort()
        codes = np.eye(len(unique_y))
        mapping = {unique_y[i]: codes[i] for i in range(len(unique_y))}
        return np.array([mapping[yi] for yi in y])

    @staticmethod
    def save(data, file_path):
        np.save(file_path, data)

    @staticmethod
    def get_randomized_indices(min_, max_):
        indices = [i for i in range(min_, max_)]
        import random
        random.shuffle(indices)
        return indices

    @staticmethod
    def load_and_save_split_data(data_set_name, parent_dir, data_path=None, x_path=None, y_path=None):
        dm = DataManager(data_path=data_path, x_path=x_path, y_path=y_path)
        dm.load()
        x_train, y_train, x_test, y_test = dm.split()
        dm.save(x_train, f'{parent_dir}/{data_set_name}xTrain.npy')
        dm.save(y_train, f'{parent_dir}/{data_set_name}yTrain.npy')
        dm.save(x_test, f'{parent_dir}/{data_set_name}xTest.npy')
        dm.save(y_test, f'{parent_dir}/{data_set_name}yTest.npy')
