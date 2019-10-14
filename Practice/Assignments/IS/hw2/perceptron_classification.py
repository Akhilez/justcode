from algorithms.perceptron import Perceptron
from data_manager import DataManager
from grapher import Grapher


def main():
    data_manager = DataManager('hw2_dataProblem.txt')
    data = data_manager.get_data()
    data = data_manager.get_column_wise_rescaled_data(data)

    grapher = Grapher()
    test_grapher = Grapher()
    fig, axs = Grapher.create_figure(1, 1, 1, figsize=(6, 4))

    x_train, y_train, x_test, y_test = data_manager.test_train_split(data, randomize=False)

    perceptron = Perceptron(x_train, y_train)
    epochs = 60
    learning_rate = 0.01

    perceptron.learn(epochs, learning_rate, x_test, y_test, grapher, test_grapher)

    predicted_ys = perceptron.test(x_test)

    hit_rate = DataManager.get_hit_rate(predicted_ys, y_test)
    print(f'Hit Rate = {hit_rate}')

    train_line, = grapher.plot(axs, label='train')
    test_line, = test_grapher.plot(axs, title='Perceptron Error', xlabel="Epochs", ylabel="Error", label='test', linewidth=4)
    grapher.plt.legend([train_line, test_line], ['Train', 'Test'])

    grapher.save_figure('figures/perceptron.png')
    grapher.show()


if __name__ == "__main__":
    main()
