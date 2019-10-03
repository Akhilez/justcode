from algorithms.knn import KNearestNeighbours
from data_manager import DataManager


def main():
    data_manager = DataManager('hw2_dataProblem.txt')
    data = data_manager.get_data()

    hit_rates = []
    for k in range(1, 15, 2):

        knn = KNearestNeighbours(k)

        for i in range(len(x)):
            train_data = data_manager.remove_rows(data, [i])
            x, y, x_test, y_test = data_manager.test_train_split(train_data, train_split_percentage=100)
            knn.load_data(x, y)
            predicted_y = knn.classify(data[i][:len(data[i])-1])
            actual_y = data[i][-1]
            hit_rate = knn.get_hit_rate(predicted_y, y_test)
            hit_rates.append(hit_rate)

    print(hit_rates)


if __name__ == "__main__":
    main()
