from algorithms.knn import KNearestNeighbours
from data_manager import DataManager


def main():
    data_manager = DataManager('hw2_dataProblem.txt')
    data = data_manager.get_data()
    x_train, y_train, x_test, y_test = data_manager.test_train_split(data)

    hit_rates = []
    for k in range(1, 15, 2):

        knn = KNearestNeighbours(k)
        knn.load_data(x_train, y_train)

        predicted_y = knn.classify(x_test)

        hit_rate = knn.get_hit_rate(predicted_y, y_test)
        hit_rates.append(hit_rate)

    print(hit_rates)


if __name__ == "__main__":
    main()
