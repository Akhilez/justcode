from akipy.neural_network import Sequential


model = Sequential.load(name='iris', parent_dir='models', find_latest=True)
