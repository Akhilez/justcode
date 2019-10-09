from algorithms.knn import KNearestNeighbours


class NeighbourhoodClassifier(KNearestNeighbours):

    def get_k_neighbours_indices(self, x):
        # TODO: Transform this method to use radius.

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
