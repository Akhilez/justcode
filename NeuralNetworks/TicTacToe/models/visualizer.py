from keras.utils import plot_model

from NeuralNetworks.TicTacToe.models.dense import DenseModel

model = DenseModel('Dense_1')
model.model_path = '/home/ak/PycharmProjects/MLBeginner/NeuralNetworks/TicTacToe/Dense_1.h5'
model.model = model.get_model()

plot_model(model.model, to_file='model.png', show_shapes=True)
