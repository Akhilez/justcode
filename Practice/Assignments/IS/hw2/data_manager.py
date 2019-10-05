class DataManager:
    def __init__(self, path_to_data_file):
        self.path = path_to_data_file

    def get_data(self):
        data = []
        with open(self.path) as file:
            lines = file.read().split('\n')[3:]
            for i, line in enumerate(lines):
                elements = line.split()
                if len(elements) > 0:
                    data.append([float(i) for i in elements])
        return data

    def test_train_split(self, data, train_split_percentage=80):
        """
        Splits a dataset randomly into test and train datasets.
        :param data: The list of lists.
        :param train_split_percentage: percentage out of 100 to be in the training set.
        :return: x_train, y_train, x_test, y_test
        """
        split_index = int(train_split_percentage/100 * len(data))

        randomized_indices = self.get_randomized_indices(0, len(data))

        x = [x[:len(x) - 1] for x in data]
        y = [i[len(i) - 1] for i in data]

        rand_x = [x[i] for i in randomized_indices]
        rand_y = [y[i] for i in randomized_indices]

        x_train = rand_x[:split_index]
        y_train = rand_y[:split_index]

        x_test = rand_x[split_index:]
        y_test = rand_y[split_index:]

        return x_train, y_train, x_test, y_test

    @staticmethod
    def get_randomized_indices(min_, max_):
        indices = [i for i in range(min_, max_)]
        import random
        random.shuffle(indices)
        return indices

    @staticmethod
    def remove_rows(data, indices):
        new_data = []
        for i in range(len(data)):
            if i not in indices:
                new_data.append(data[i])
        return new_data
