from algorithms.knn import KNearestNeighbours
from algorithms.perceptron import Perceptron
from data_manager import DataManager


def main():
    data_manager = DataManager('hw2_dataProblem.txt')
    data = data_manager.get_data()

    grapher = KNearestNeighbours.Grapher()
    fig, axs = grapher.create_figure(1, 1, 1, figsize=(6, 4))

    x_train, y_train, x_test, y_test = data_manager.test_train_split(data)

    perceptron = Perceptron(x_train, y_train)
    epochs = 1000
    learning_rate = 0.0000001

    perceptron.learn(epochs, learning_rate)

    predicted_ys = perceptron.test(x_test)

    hit_rate = KNearestNeighbours.get_hit_rate(predicted_ys, y_test)
    print(f'Hit Rate = {hit_rate}')

    grapher.record(hit_rate, [1])

    grapher.plot(axs, title='Perceptron', xlabel="N", ylabel="Hit-Rate")
    grapher.save_figure('figures/perceptron.png')
    grapher.show()


if __name__ == "__main__":
    main()
