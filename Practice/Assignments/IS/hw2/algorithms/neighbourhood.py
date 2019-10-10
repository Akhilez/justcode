from algorithms.knn import KNearestNeighbours


class NeighbourhoodClassifier(KNearestNeighbours):

    def get_k_neighbours_indices(self, x):

        radius = self.k

        # From x, create a radius of x - radius and x + radius
        ranges = [range(int(x[i] - radius/2), int(x[i] + radius/2)) for i in range(len(x))]

        # Find all the points in this radius.
        points_in_ranges_indices = [i for i in range(len(self.x)) if
                                    all([int(self.x[i][j]) in ranges[j] for j in range(len(ranges))])]

        # Find the distances of all points in the square
        distances = [self.find_distance(x, self.x[i]) for i in points_in_ranges_indices]

        distances = [distance for distance in distances if distance <= radius]

        # print(f"Points = {len(distances)}")

        # Pick the shortest k distances and return their indices
        min_distance_indices = []
        distances_copy = list(distances)
        for i in range(len(distances)):
            min_distance_index = distances_copy.index(min(distances_copy))
            min_distance_indices.append(min_distance_index)
            del distances_copy[min_distance_index]
            i += 1

        return [points_in_ranges_indices[i] for i in min_distance_indices]
