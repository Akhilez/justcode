from data_manager import DataManager
from layers import Input, Dense
from neural_network import Sequential

dm = DataManager(data_path='data/iris.data')
dm.load(split=True, one_hot=True)

model = Sequential()

model.add(Input(units=4))
model.add(Dense(units=10, activation='sigmoid'))
model.add(Dense(units=10, activation='sigmoid'))
model.add(Dense(units=3, activation='sigmoid'))

model.compile(optimizer='SGD', loss='MSE', metrics=['every_tenth_hit_rate'])

model.train(dm.x_train, dm.y_train, validation_set=(dm.x_test, dm.y_test), epochs=200, lr=0.5, momentum=0.01)

y_pred = model.test(dm.x_test)
y_pred = dm.get_winner_take_all(y_pred)

confusion_matrix = dm.get_confusion_matrix(dm.y_test, y_pred)
print(f'Confusion matrix: \n{confusion_matrix}')

# model.save('test', parent_dir='models')

# model = model.load('test', parent_dir='models')
