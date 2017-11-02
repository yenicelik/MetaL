import numpy as np
from sklearn import metrics


class Losses:

    def __init__(self):
        self.loss = None
        self.loss_set = False
        self.ys = []
        self.pred_ys = []

        self.y = None
        self.pred_y = None

    def add_data(self, new_y, new_pred_y):
        self.ys.extend(np.asarray(new_y))
        self.pred_ys.extend(np.asarray(new_pred_y))

    def _prepare_data(self):
        self.y = np.array(self.ys)
        self.pred_y = np.array(self.pred_ys)

    # Actual Loss Functions
    def mean_squared_error(self):
        self._prepare_data()
        out = metrics.mean_squared_error(self.y, self.pred_y)
        return out
