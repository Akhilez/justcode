from losses import get_loss_function
from optimizers import get_optimizer


class Sequential:
    name = 'sequential'

    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.metrics_names = None
        self.optimizer = None

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

    def test(self, x_test):
        y_pred = []
        for xq in x_test:
            metrics = self.optimizer.feed(xq)
            y_pred.append(metrics['y_pred'])
        return y_pred

    def save(self, model_name, parent_dir):
        from datetime import datetime
        import json
        model = self.get_structure()
        timestamp = str(datetime.now().timestamp()).split('.')[0]
        with open(f'{parent_dir}/{model_name}_{timestamp}.json', 'w', encoding='utf-8') as file:
            json.dump(model, file)

    def get_structure(self):
        return {
            'name': self.name,
            'layers': [layer.get_serialized() for layer in self.layers],
            'loss': self.loss_function.name,
            'metrics': self.metrics_names
        }

    def describe(self):
        print(self.get_structure())

    def load(self, name, parent_dir, find_latest=False):
        import json
        model_name = name if not find_latest else self._get_latest_model_name(name, parent_dir)
        with open(f'{parent_dir}/{model_name}', 'w', encoding='utf-8') as json_file:
            structure = json.load(json_file)
            return self._create_model_from_structure(structure)

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
        model.loss_function = get_loss_function(structure['loss'])
        model.metrics_names = structure['metrics']

        import layers
        model.layers = [layers.create_layer_from_structure(layer) for layer in structure['layers']]


def get_neural_network(name):
    if name == Sequential.name:
        return Sequential()
    else:
        raise Exception(f"Model not found for {name}")


class Metrics:
    ERROR = 'error'
    ACCURACY = 'accuracy'
    HIT_RATE = 'hit-rate'
    EVERY_TENTH_HIT_RATE = 'every_tenth_hit_rate'

    def __init__(self, to_collect, model=None):
        self.to_collect = to_collect
        self.model = model

        # Each epoch
        self.current_epoch = -1
        self.errors = []
        self._epoch_error = -1
        self.hit_rates = []

        # Each iteration
        self.current_iteration = -1
        self.iter_error = []
        self._iter_y_train = []
        self._iter_y_preds = []

        # Every 10th epoch
        self.every_tenth_indices = []
        self.every_tenth_error = []
        self.tenth_epochs_hit_rates = []

    def _collect_tenth_epoch_metrics(self):
        self.every_tenth_error.append(self._epoch_error)
        self.every_tenth_indices.append(self.current_epoch)

    def collect_iteration_metrics(self, xq, yq, yh=None):
        self.current_iteration += 1
        error = self.model.loss_function.f(yq, yh)
        self.iter_error.append(error)
        self._iter_y_preds.append(yh)
        self._iter_y_train.append(yq)

    def collect_post_epoch(self, x_train=None, y_train=None, validation_set=None):
        self.current_epoch += 1

        self._epoch_error = sum(self.iter_error)

        if self.ERROR in self.to_collect:
            self.errors.append(self._epoch_error)

        if self.EVERY_TENTH_HIT_RATE in self.to_collect:
            if self.current_epoch == 0 or self.current_epoch % 10 == 0:
                self._collect_tenth_epoch_metrics()

        # TODO: Implement metrics for validation set.
        # TODO: Calculate hit_rate

        self._clear_post_epoch()

    def _clear_post_epoch(self):
        self.iter_error.clear()
        self.current_iteration = -1
        self._iter_y_train.clear()
        self._iter_y_preds.clear()
        self._epoch_error = -1


