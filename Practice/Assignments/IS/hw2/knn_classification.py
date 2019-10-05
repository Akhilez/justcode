from algorithms.knn import KNearestNeighbours
from data_manager import DataManager


def main():
    data_manager = DataManager('hw2_dataProblem.txt')
    data = data_manager.get_data()

    actual_ys = []
    predicted_ys = []
    for k in range(1, 15, 2):

        knn = KNearestNeighbours(k)

        for i in range(len(data)):
            train_data = data_manager.remove_rows(data, [i])
            x, y, x_test, y_test = data_manager.test_train_split(train_data, train_split_percentage=100)
            knn.load_data(x, y)

            predicted_y = knn.classify([data[i][:len(data[i])-1]])
            predicted_ys.append(predicted_y)

            actual_y = [data[i][-1]]
            actual_ys.append(actual_y)

    hit_rate = KNearestNeighbours.get_hit_rate(predicted_ys, actual_ys)
    print(hit_rate)


if __name__ == "__main__":
    main()