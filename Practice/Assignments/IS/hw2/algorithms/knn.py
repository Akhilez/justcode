import matplotlib.pyplot as plt


class KNearestNeighbours:
    def __init__(self, k):
        self.k = k
        self.x = None
        self.y = None

    def load_data(self, x, y):
        self.x = x
        self.y = y

    def classify(self, xs):
        classes = []
        for x in xs:
            closest_neighbours = self.get_closest_neighbouring_classes(x)
            best_class = self.get_best_class(neighbours=closest_neighbours)
            classes.append(best_class)
        return classes

    @staticmethod
    def get_hit_rate(predicted, actual):
        hits = 0
        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                hits += 1
        return hits / len(predicted)

    def get_closest_neighbouring_classes(self, x):
        k_neighbours_indices = self.get_k_neighbours_indices(x)
        classes = [self.y[i] for i in k_neighbours_indices]
        return classes

    def get_best_class(self, neighbours):
        if len(neighbours) == 0:
            return
        return max(neighbours, key=neighbours.count)

    def get_k_neighbours_indices(self, x):

        # Find the standard deviation of the data.
        standard_deviations = self.get_standard_deviations()

        radius = 1

        while True:

            # From x, create a radius of x - std and x + std
            ranges = [range(int(x[i] - radius * standard_deviations[i]), int(x[i] + radius * standard_deviations[i])) for i in range(len(standard_deviations))]

            # Find all the points in this radius.
            points_in_ranges_indices = [i for i in range(len(self.x)) if
                                        all([int(self.x[i][j]) in ranges[j] for j in range(len(ranges))])]

            # If these points are more than k, stop.
            if len(points_in_ranges_indices) > self.k:
                break
            # Else, radius += std. Find all now.
            else:
                radius += 1

        distances = [self.find_distance(x, self.x[i]) for i in points_in_ranges_indices]

        # Pick the shortest k distances and return their indices
        min_distance_indices = []
        distances_copy = list(distances)
        for i in range(self.k):
            min_distance_index = distances_copy.index(min(distances_copy))
            min_distance_indices.append(min_distance_index)
            del distances_copy[min_distance_index]
            i += 1

        return [points_in_ranges_indices[i] for i in min_distance_indices]

    def get_standard_deviations(self):
        # Find std for each attribute

        if len(self.x) == 0:
            return

        stds = []

        for i in range(len(self.x[0])):
            column = [j[i] for j in self.x]
            avg = sum(column)//len(column)
            std = (sum([(avg - i) ** 2 for i in column]) // len(column)) ** 0.5
            stds.append(int(std))

        return stds

    @staticmethod
    def find_distance(x1, x2):
        # Find euclidean distance between these two points.
        distance = 0
        for i in range(len(x1)):
            distance += (x1[i] - x2[i]) ** 2
        distance = distance ** 0.5
        return distance

    class Grapher:
        """
        The whole purpose of this class is just to create some subplots and store the x and y data.
        """

        def __init__(self):
            self.x = []
            self.y = []

        def record(self, x_value, y_value):
            self.x.append(x_value)
            self.y.append(y_value)

        def plot(self, axis, title='Title', xlabel="x", ylabel="y", ylim=None, percentage_from_last=100, xticks=None):
            starting_index = int((1 - percentage_from_last / 100) * len(self.x))
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            if ylim:
                axis.set_ylim(ylim)
            axis.set_title(title)
            if xticks:
                axis.set_xticks(xticks)
            axis.plot(self.x[starting_index:], self.y[starting_index:])

        def clear_data(self):
            self.__init__()

        @staticmethod
        def create_figure(num_rows, num_columns, figure_number, figsize=(16, 10)):
            return plt.subplots(num_rows, num_columns, constrained_layout=True, num=figure_number, figsize=figsize,
                                dpi=80)

        @staticmethod
        def show():
            plt.show()

        @staticmethod
        def save_figure(path):
            plt.savefig(path)
