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

    def test_train_split(self, data, train_split_percentage=80, randomize=True):
        """
        Splits a dataset randomly into test and train datasets.
        :param randomize: Will randomize the data points.
        :param data: The list of lists.
        :param train_split_percentage: percentage out of 100 to be in the training set.
        :return: x_train, y_train, x_test, y_test
        """
        split_index = int(train_split_percentage / 100 * len(data))

        indices = self.get_randomized_indices(0, len(data)) if randomize else [i for i in range(len(data))]

        x = [x[:len(x) - 1] for x in data]
        y = [i[len(i) - 1] for i in data]

        rand_x = [x[i] for i in indices]
        rand_y = [y[i] for i in indices]

        x_train = rand_x[:split_index]
        y_train = rand_y[:split_index]

        x_test = rand_x[split_index:]
        y_test = rand_y[split_index:]

        return x_train, y_train, x_test, y_test

    @staticmethod
    def get_randomized_indices(min_, max_):
        indices = [i for i in range(min_, max_)]
        import random
        random.seed(2)
        random.shuffle(indices)
        return indices

    @staticmethod
    def remove_rows(data, indices):
        new_data = []
        for i in range(len(data)):
            if i not in indices:
                new_data.append(data[i])
        return new_data

    @staticmethod
    def get_hit_rate(predicted, actual):
        hits = 0
        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                hits += 1
        return hits / len(predicted)

    @staticmethod
    def get_column_wise_rescaled_data(data):
        """
        For each column
            find min and max
            period = max - min
            for each point in column
                x_new = (x-min) / period
        :param data: The unscaled data
        :return: Scaled data
        """
        if len(data) == 0:
            return data
        new_data = list(data)
        for i in range(len(data[0])):
            column = [x[i] for x in data]
            min_ = min(column)
            max_ = max(column)
            period = max_ - min_

            for j in range(len(column)):
                new_data[j][i] = (data[j][i] - min_) / period

        return new_data
