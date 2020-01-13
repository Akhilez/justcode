from utils.data_manager import DataManager

DataManager.load_and_save_split_data(x_path='data/MNISTnumImages5000.txt', y_path='data/MNISTnumLabels5000.txt', data_set_name='Mnist', parent_dir='data')
