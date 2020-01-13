from data_manager import DataManager
import matplotlib.pyplot as plt


def main():
    data_manager = DataManager('hw2_dataProblem.txt')
    data = data_manager.get_data()
    data = data_manager.get_column_wise_rescaled_data(data)

    ones = [x[:2] for x in data if x[2] == 1]
    zeros = [x[:2] for x in data if x[2] == 0]

    plt.figure(figsize=(4, 4), constrained_layout=True)

    plt.scatter([x[0] for x in ones], [x[1] for x in ones], s=10)
    plt.scatter([x[0] for x in zeros], [x[1] for x in zeros], s=10)

    plt.title("Data visualization")
    plt.xlabel("L")
    plt.ylabel("P")
    plt.legend(['1', '0'])

    plt.savefig('figures/scatter.png')
    plt.show()


if __name__ == "__main__":
    main()
