import numpy as np
from utils.logger import logger


class Metrics:
    ERROR = 'error'
    ACCURACY = 'accuracy'
    HIT_RATE = 'hit-rate'
    CLASSIFICATION_ERROR = 'classification_error'
    EVERY_TENTH_HIT_RATE = 'every_tenth_hit_rate'
    EVERY_TENTH_CLASSIFICATION_ERROR = 'every_tenth_classification_error'
    EVERY_TENTH_ERROR = 'every_tenth_error'
    ITER_ERRORS = 'iter_errors'

    def __init__(self, to_collect, model=None):
        self.to_collect = to_collect
        self.model = model

        # Each epoch
        self.current_epoch = -1
        self.errors = []
        self._epoch_error = -1
        self.hit_rates = []
        self.print_message = []
        self._epoch_validation_error = -1
        self._epoch_validation_y_pred = None

        # Each iteration
        self.current_iteration = -1
        self.iter_error = []
        self.iter_errors = []
        self._iter_x_train = []
        self._iter_y_train = []
        self._iter_y_preds = []
        self._hit_rate = None

        # Every 10th epoch
        self.tenth_epoch_indices = []
        self.tenth_epoch_errors = []
        self.tenth_epoch_hit_rates = []
        self.tenth_epoch_validation_hit_rates = []
        self.tenth_epoch_classification_error = []

    def _collect_tenth_epoch_metrics(self, validation_set=None):
        self.tenth_epoch_indices.append(self.current_epoch)
        if self.EVERY_TENTH_ERROR in self.to_collect:
            self.tenth_epoch_errors.append(self._epoch_error)
            self.print_message.append(f'Tenth Error: {self._epoch_error}')
        if self.EVERY_TENTH_HIT_RATE in self.to_collect:
            self.tenth_epoch_hit_rates.append(self._get_hit_rate())
            self.print_message.append(f'HitRate: {self._get_hit_rate()}')
            if validation_set is not None:
                validation_set_tenth_hit_rate = self.get_hit_rate(
                    self.get_winner_take_all(self._epoch_validation_y_pred), validation_set[1])
                self.tenth_epoch_validation_hit_rates.append(validation_set_tenth_hit_rate)
                self.print_message.append(f'Validation Hit-Rate: {validation_set_tenth_hit_rate}')
        if self.EVERY_TENTH_CLASSIFICATION_ERROR in self.to_collect:
            self.tenth_epoch_classification_error.append(1 - self._get_hit_rate())

    def _get_hit_rate(self):
        if self._hit_rate is not None:
            return self._hit_rate
        y_preds_winners = self.get_winner_take_all(self._iter_y_preds)
        self._hit_rate = self.get_hit_rate(y_preds_winners, self._iter_y_train)
        return self._hit_rate

    def collect_iteration_metrics(self, xq, yq, yh=None):
        self.current_iteration += 1
        error = self.model.loss_function.f(yq, yh)
        self.iter_error.append(error)
        self._iter_x_train.append(xq)
        self._iter_y_train.append(yq)
        self._iter_y_preds.append(yh)

    def collect_post_epoch(self, validation_set=None):
        self.current_epoch += 1
        self.print_message.append(f'Epoch: {self.current_epoch}')

        self._epoch_error = sum(self.iter_error)/len(self.iter_error)
        if validation_set is not None:
            self._set_validation_error(validation_set)

        if self.ERROR in self.to_collect:
            self.errors.append(self._epoch_error)
            self.print_message.append(f'Error: {self._epoch_error}')
            if validation_set is not None:
                self.print_message.append(f'Validation Error: {self._epoch_validation_error}')

        if self.current_epoch % 10 == 0:
            self._collect_tenth_epoch_metrics(validation_set)

        if self.HIT_RATE in self.to_collect:
            self.hit_rates.append(self._get_hit_rate())

        if self.ITER_ERRORS in self.to_collect:
            self.iter_errors.append(list(self.iter_error))

        # TODO: Implement metrics for validation set.

        if len(self.print_message) > 0:
            logger.info(' | '.join(self.print_message))
        self._clear_post_epoch()

    def _set_validation_error(self, validation_set):
        x_test, y_test = validation_set
        self._epoch_validation_y_pred = self.model.test(x_test)
        errors = sum(self.model.loss_function.f(y_test, self._epoch_validation_y_pred).T)
        self._epoch_validation_error = sum(errors)/len(errors)

    def _clear_post_epoch(self):
        self.iter_error.clear()
        self.current_iteration = -1
        self._iter_y_train.clear()
        self._iter_y_preds.clear()
        self._iter_x_train.clear()
        self._epoch_error = -1
        self._epoch_validation_error = -1
        self._hit_rate = None
        self.print_message.clear()
        self._epoch_validation_y_pred = None

    @staticmethod
    def get_hit_rate(predicted, actual):
        hits = 0
        for i in range(len(predicted)):
            if all(predicted[i] == actual[i]):
                hits += 1
        return hits / len(predicted)

    @staticmethod
    def get_winner_take_all(y):
        winner_y = []
        for yq in y:
            max_index = yq.argmax()
            winner_y.append([1 if i == max_index else 0 for i in range(len(yq))])
        return np.array(winner_y)

    @staticmethod
    def get_confusion_matrix(y_test, y_pred):
        (n_rows, n_cols) = y_test.shape
        classes = np.unique(y_test, axis=0)
        matrix = np.zeros((n_cols, n_cols))
        for i in range(n_rows):
            desired_index = [j for j in range(n_cols) if all(y_test[i] == classes[j])][0]
            pred_index = [j for j in range(n_cols) if all(y_pred[i] == classes[j])][0]
            matrix[pred_index][desired_index] += 1
        return matrix
