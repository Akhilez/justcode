from data_manager import DataManager
from layers import Input, Dense
from neural_network import Sequential

dm = DataManager(data_path='data/iris.data')
dm.load(split=True, one_hot=True)

model = Sequential()

model.add(Input(units=4))
model.add(Dense(units=10, activation='sigmoid'))
model.add(Dense(units=3, activation='sigmoid'))

model.compile(optimizer='SGD', loss='MSE', metrics=['accuracy', 'error'])

model.train(dm.x_train, dm.y_train, validation_set=(dm.x_test, dm.y_test), epochs=200, lr=0.5)

y_pred = model.test(dm.x_test)

for i in range(len(y_pred)):
    print(f'{dm.y_test[i]} %%%% {y_pred[i]}')

model.save('test', parent_dir='models')

# model = model.load('test', parent_dir='models')
