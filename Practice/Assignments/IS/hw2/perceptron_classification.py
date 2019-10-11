from algorithms.knn import KNearestNeighbours
from algorithms.perceptron import Perceptron
from data_manager import DataManager


def main():
    data_manager = DataManager('hw2_dataProblem.txt')
    data = data_manager.get_data()
    data = data_manager.get_column_wise_rescaled_data(data)

    grapher = KNearestNeighbours.Grapher()
    fig, axs = grapher.create_figure(1, 1, 1, figsize=(6, 4))

    x_train, y_train, x_test, y_test = data_manager.test_train_split(data)

    perceptron = Perceptron(x_train, y_train)
    epochs = 10
    learning_rate = 0.1

    perceptron.learn(epochs, learning_rate, grapher)

    predicted_ys = perceptron.test(x_test)

    for i in range(len(y_test)):
        print(f'Real = {y_test[i]}, predicted = {predicted_ys[i]}, diff = {y_test[i] - predicted_ys[i]}')

    hit_rate = KNearestNeighbours.get_hit_rate(predicted_ys, y_test)
    print(f'Hit Rate = {hit_rate}')

    grapher.plot(axs, title='Perceptron', xlabel="Epochs", ylabel="Error")
    grapher.save_figure('figures/perceptron.png')
    grapher.show()


if __name__ == "__main__":
    main()
