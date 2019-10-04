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
        return hits/len(predicted)

    def get_closest_neighbouring_classes(self, x):
        k_neighbours_indices = self.get_k_neighbours_indices(x)
        classes = [self.y[i] for i in k_neighbours_indices]
        return classes

    def get_best_class(self, neighbours):
        return self.mode(neighbours)

    @staticmethod
    def mode(elements):
        counts = {}
        for neighbour in elements:
            if counts.get(neighbour) is None:
                count = 0
                for i in elements:
                    if neighbour == i:
                        count += 1
                counts[neighbour] = count

        return max(counts.values())

    def get_k_neighbours_indices(self, x):
        # TODO: Find k closest neighbours to x

        # Find the standard deviation of the data.
        standard_deviations = self.get_standard_deviations()

        # From x, create a radius of x - std and x + std
        x_range = range(x[0])

        # Find all the points in this radius. If these points are more than k, stop.
        # Else, radius += std. Find all now.
        # Find the distances of all the points in this radius.
        # Pick the shortest k distances and return their indices

    def get_standard_deviations(self):
        # Find std for each attribute

        # avg_x = sum(self.x)
        # std_x = (sum([(avg_x - i) ** 2 for i in self.x]) / len(self.x)) ** 0.5

        # return std_x
        pass
