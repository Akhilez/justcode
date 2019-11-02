from losses import get_loss_function


class Sequential:
    name = 'sequential'

    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.metrics_names = None

    def add(self, layer):
        if len(layer) > 0:
            prev_layer = self.layers[-1]
            layer.set_input_size(prev_layer.n_units)
        self.layers.append(layer)

    def compile(self, loss, metrics):
        self.loss_function = get_loss_function(loss)
        self.metrics_names = metrics

    def train(self, x_train, y_train, epochs, lr, validation_set=None):
        # TODO: Implement method
        pass

    def save(self, model_name, parent_dir):
        from datetime import datetime
        import json
        model = {
            'name': self.name,
            'layers': [layer.get_serialized() for layer in self.layers],
            'loss': self.loss_function.name,
            'metrics': self.metrics_names
        }
        timestamp = datetime.now()
        with open(f'{parent_dir}/{model_name}_{timestamp}.json', 'w', encoding='utf-8') as file:
            json.dump(model, file)

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
