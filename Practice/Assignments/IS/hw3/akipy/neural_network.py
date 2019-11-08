from akipy.losses import get_loss_function
from akipy.metrics import Metrics
from akipy.optimizers import get_optimizer

import numpy as np


class Sequential:
    name = 'sequential'

    def __init__(self, name='sequential'):
        self.layers = []
        self.loss_function = None
        self.metrics_names = None
        self.optimizer = None
        self.model_name = name

    def add(self, layer):
        if len(self.layers) > 0:
            prev_layer = self.layers[-1]
            layer.set_input_size(prev_layer.n_units)
        self.layers.append(layer)

    def compile(self, optimizer, loss, metrics):
        self.loss_function = get_loss_function(loss)
        self.metrics_names = metrics
        self.optimizer = get_optimizer(optimizer, model=self)

    def train(self, x_train, y_train, epochs, lr, validation_set=None, momentum=None):
        # Input size validation
        if x_train.shape[1] != self.layers[0].n_units:
            raise Exception(f"Input shape {x_train.shape} does not match with input layer {self.layers[0].n_units}.")

        self.optimizer.set_training_data(x_train, y_train, lr, validation_set, momentum)
        metrics = Metrics(self.metrics_names, model=self)

        # Start epochs
        for epoch in range(epochs):
            for i in range(len(x_train)):
                xq, yq = self.optimizer.get_data_point()
                self.optimizer.feed(xq, yq, metrics=metrics)

            metrics.collect_post_epoch(x_train=x_train, y_train=y_train, validation_set=validation_set)

        return metrics

    def test(self, x_test, y_test=None):

        metrics = None if y_test is None else Metrics(self.metrics_names, model=self)

        y_pred = []
        for q in range(len(x_test)):
            yq = None if y_test is None else y_test[q]
            y_pred_i = self.optimizer.feed(x_test[q], yq=yq, metrics=metrics, learn=False)
            y_pred.append(y_pred_i)

        if y_test is not None:
            metrics.collect_post_epoch()
            return np.array(y_pred), metrics

        return np.array(y_pred)

    def save(self, parent_dir):
        model = self.get_structure()
        with open(f'{parent_dir}/{self.model_name}_{model["timestamp"]}.json', 'w', encoding='utf-8') as file:
            import json
            json.dump(model, file)

    def get_structure(self):
        from datetime import datetime
        return {
            'name': self.name,
            'model_name': self.model_name,
            'layers': [layer.get_serialized() for layer in self.layers],
            'loss': self.loss_function.name,
            'optimizer': self.optimizer.name,
            'metrics': self.metrics_names,
            'timestamp': str(datetime.now().timestamp()).split('.')[0]
        }

    def describe(self):
        print(self.get_structure())

    @staticmethod
    def load(name, parent_dir, find_latest=False):
        model_name = name if not find_latest else Sequential._get_latest_model_name(name, parent_dir)
        with open(f'{parent_dir}/{model_name}', 'r', encoding='utf-8') as json_file:
            import json
            structure = json.load(json_file)
            return Sequential._create_model_from_structure(structure)

    @staticmethod
    def _get_latest_model_name(model_name, parent_dir):
        import os
        files = os.listdir(parent_dir)
        files.sort(reverse=True)
        for file in files:
            if model_name in file:
                return file

    @staticmethod
    def _create_model_from_structure(structure):
        model = get_neural_network(structure['name'])
        model.model_name = structure['model_name']
        model.loss_function = get_loss_function(structure['loss'])
        model.metrics_names = structure['metrics']
        model.optimizer = get_optimizer(structure['optimizer'], model=model)

        from akipy import layers
        model.layers = [layers.create_layer_from_structure(layer) for layer in structure['layers']]

        return model


def get_neural_network(name):
    if name == Sequential.name:
        return Sequential()
    else:
        raise Exception(f"Model not found for {name}")

