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

        # Find the standard deviation of the data.
        standard_deviations = self.get_standard_deviations()

        radius = 1

        while True:

            # From x, create a radius of x - std and x + std
            ranges = [range(x[i] - radius * standard_deviations, x[i] + radius * standard_deviations) for i in
                      range(len(standard_deviations))]

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
        for i in range(self.k - 1):
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

        for i in range(self.x[0]):
            column = [j[i] for j in self.x]
            avg = sum(column)
            std = (sum([(avg - i) ** 2 for i in column]) / len(column)) ** 0.5
            stds.append(std)

        return stds

    def find_distance(self, x1, x2):
        # TODO: Find euclidian distance between these two points.
        return 0
