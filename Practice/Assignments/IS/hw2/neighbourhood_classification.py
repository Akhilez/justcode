from algorithms.neighbourhood import NeighbourhoodClassifier
from data_manager import DataManager


def main():
    data_manager = DataManager('hw2_dataProblem.txt')
    data = data_manager.get_data()
    grapher = NeighbourhoodClassifier.Grapher()
    fig, axs = grapher.create_figure(1, 1, 1, figsize=(6, 4))

    for radius in range(5, 20):

        print(f'Radius = {radius}')

        actual_ys = []
        predicted_ys = []

        knn = NeighbourhoodClassifier(radius)

        for i in range(len(data)):
            train_data = data_manager.remove_rows(data, [i])
            x, y, x_test, y_test = data_manager.test_train_split(train_data, train_split_percentage=100, randomize=False)
            knn.load_data(x, y)

            predicted_y = knn.classify([data[i][:len(data[i])-1]])
            predicted_ys.append(predicted_y)

            actual_y = [data[i][-1]]
            actual_ys.append(actual_y)

        hit_rate = NeighbourhoodClassifier.get_hit_rate(predicted_ys, actual_ys)
        print(f'Hit Rate = {hit_rate}')

        grapher.record(radius, hit_rate)

    grapher.plot(axs, title='Neighbourhood Classification: Radius vs Hit-Rate', xlabel="Radius", ylabel="Hit-Rate", xticks=grapher.x)
    grapher.save_figure('figures/neighbourhood.png')
    grapher.show()


if __name__ == "__main__":
    main()
