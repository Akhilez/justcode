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
