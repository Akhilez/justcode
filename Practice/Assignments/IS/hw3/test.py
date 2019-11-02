from data_manager import DataManager
from layers import Input, Dense
from neural_network import Sequential

dm = DataManager(data_path='data/iris.data')
dm.load(split=True, one_hot=True)

model = Sequential()

model.add(Input(units=4))
model.add(Dense(units=5, activation='sigmoid'))
model.add(Dense(units=3, activation='sigmoid'))

model.compile(optimizer='SGD', loss='MSE', metrics=['accuracy', 'error'])

model.train(dm.x_train, dm.y_train, validation_set=(dm.x_test, dm.y_test), epochs=10, lr=0.00001)

model.save('test', parent_dir='models')

model = model.load('test', parent_dir='models')
