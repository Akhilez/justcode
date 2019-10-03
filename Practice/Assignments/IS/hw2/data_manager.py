class DataManager:
    def __init__(self, path_to_data_file):
        self.path = path_to_data_file

    def get_data(self):
        return [[1, 2, 9], [3, 4, 8], [5, 6, 9]]
        # TODO: Get the data from the data file.

    def test_train_split(self, data, train_split_percentage=80):
        """
        Splits a dataset randomly into test and train datasets.
        :param data: The list of lists.
        :param train_split_percentage: percentage out of 100 to be in the training set.
        :return: x_train, y_train, x_test, y_test
        """
        split_index = int(train_split_percentage/100 * len(data))

        randomized_indices = self.get_randomized_indices(0, len(data))

        x = [x[:len(data) - 1] for x in data]
        y = [x[len(data) - 1] for x in data]

        rand_x = [x[i] for i in randomized_indices]
        rand_y = [y[i] for i in randomized_indices]

        x_train = rand_x[:split_index]
        y_train = rand_y[:split_index]

        x_test = rand_x[split_index:]
        y_test = rand_y[split_index:]

        return x_train, y_train, x_test, y_test

    @staticmethod
    def get_randomized_indices(min_, max_):
        # TODO: Randomize the indices
        return [x for x in range(min_, max_)]

    @staticmethod
    def remove_rows(data, indices):
        new_data = []
        for i in range(len(data)):
            if i not in indices:
                new_data.append(data[i])
        return new_data
