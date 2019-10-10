from data_manager import DataManager


def main():
    data_manager = DataManager('hw2_dataProblem.txt')
    data = data_manager.get_data()
    data = data_manager.get_rescaled_data(data, data_manager.get_scaling_function(data))




if __name__ == "__main__":
    main()
