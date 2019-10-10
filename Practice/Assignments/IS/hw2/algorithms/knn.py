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
        k_neighbours_indices = self.get_close_neighbours_indices(x)
        classes = [self.y[i] for i in k_neighbours_indices]
        return classes

    def get_best_class(self, neighbours):
        if len(neighbours) == 0:
            return
        return max(neighbours, key=neighbours.count)

    def get_close_neighbours_indices(self, x):
        distances = [self.find_distance(x, xi) for xi in self.x]
        sorted_distances = sorted(range(len(distances)), key=lambda k: distances[k])[:self.k]
        return sorted_distances

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
