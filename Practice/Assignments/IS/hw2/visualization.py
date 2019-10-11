from data_manager import DataManager
import matplotlib.pyplot as plt


def main():
    data_manager = DataManager('hw2_dataProblem.txt')
    data = data_manager.get_data()
    data = data_manager.get_column_wise_rescaled_data(data)

    ones = [x[:2] for x in data if x[2] == 1]
    zeros = [x[:2] for x in data if x[2] == 0]

    plt.scatter([x[0] for x in ones], [x[1] for x in ones])
    plt.scatter([x[0] for x in zeros], [x[1] for x in zeros])

    # plt.savefig('figures/scatter.png')
    plt.show()


if __name__ == "__main__":
    main()
