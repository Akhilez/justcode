import numpy as np


class Metrics:
    ERROR = 'error'
    ACCURACY = 'accuracy'
    HIT_RATE = 'hit-rate'
    CLASSIFICATION_ERROR = 'classification_error'
    EVERY_TENTH_HIT_RATE = 'every_tenth_hit_rate'
    EVERY_TENTH_CLASSIFICATION_ERROR = 'every_tenth_classification_error'

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
        self._iter_x_train = []
        self._iter_y_train = []
        self._iter_y_preds = []
        self._hit_rate = None

        # Every 10th epoch
        self.tenth_epoch_indices = []
        self.tenth_epoch_errors = []
        self.tenth_epoch_hit_rates = []
        self.tenth_epoch_classification_error = []

    def _collect_tenth_epoch_metrics(self):
        if self.EVERY_TENTH_HIT_RATE in self.to_collect:
            self.tenth_epoch_indices.append(self.current_epoch)
            if self.ERROR in self.to_collect:
                self.tenth_epoch_errors.append(self._epoch_error)
            if self.EVERY_TENTH_HIT_RATE in self.to_collect:
                self.tenth_epoch_hit_rates.append(self._get_hit_rate())
            if self.EVERY_TENTH_CLASSIFICATION_ERROR in self.to_collect:
                self.tenth_epoch_classification_error.append(1 - self._get_hit_rate())

    def _get_hit_rate(self):
        if self._hit_rate is not None:
            return self._hit_rate
        hits = 0
        y_preds_winners = self.get_winner_take_all(self._iter_y_preds)
        for yi in range(len(y_preds_winners)):
            if all(self._iter_y_train[yi] == y_preds_winners[yi]):
                hits += 1
        self._hit_rate = hits / len(y_preds_winners)
        return self._hit_rate

    def collect_iteration_metrics(self, xq, yq, yh=None):
        self.current_iteration += 1
        error = self.model.loss_function.f(yq, yh)
        self.iter_error.append(error)
        self._iter_x_train.append(xq)
        self._iter_y_train.append(yq)
        self._iter_y_preds.append(yh)

    def collect_post_epoch(self, x_train=None, y_train=None, validation_set=None):
        self.current_epoch += 1

        self._epoch_error = sum(self.iter_error)

        if self.ERROR in self.to_collect:
            self.errors.append(self._epoch_error)

        if self.current_epoch == 0 or self.current_epoch % 10 == 0:
            self._collect_tenth_epoch_metrics()

        if self.HIT_RATE in self.to_collect:
            self.hit_rates.append(self._get_hit_rate())

        # TODO: Implement metrics for validation set.

        self._clear_post_epoch()

    def _clear_post_epoch(self):
        self.iter_error.clear()
        self.current_iteration = -1
        self._iter_y_train.clear()
        self._iter_y_preds.clear()
        self._iter_x_train.clear()
        self._epoch_error = -1
        self._hit_rate = None

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
