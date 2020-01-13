from algorithms.knn import KNearestNeighbours


class NeighbourhoodClassifier(KNearestNeighbours):

    def get_close_neighbours_indices(self, x):
        distances = [self.find_distance(x, xi) for xi in self.x]
        sorted_distances_indices = sorted(range(len(distances)), key=lambda k: distances[k])
        points_within_k = [index for index in sorted_distances_indices if distances[index] <= self.k]
        return points_within_k
