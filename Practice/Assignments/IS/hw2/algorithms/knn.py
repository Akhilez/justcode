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
        neighbours = []
        classes = []
        # TODO: Find k closest neighbours to x
        # TODO: Get the class of the neighbour
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
