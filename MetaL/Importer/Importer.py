import pandas as pd
import numpy as np


class Importer:
    """ This class viszalizes any 3D object """

    def __init__(self, X_path, y_path=None, dev=False):
        self.RND_SEED = 0  # as we want the experiments to be reproducable!
        if y_path is None:
            self.X, _ = self._import_data(X_path, y_path)
            self.y = None
        else:
            self.X, self.y = self._import_data(X_path, y_path)
        self.dev = dev
        self.X_path = X_path
        self.y_path = y_path
 
        # Hyperparameters
        self.dev_set_size = 5  # int(self.X.shape[0]*0.1) # **0.7 / **0.8 good values

        if dev:
            self.X, self.y = self._prepare_dev_set()  # is X and y is returned always applicable

    def getX(self):
        return self.X

    def gety(self):
        return self.y

    def _import_data(self, X_path, y_path=None):
        # Currently assume that we have pandas-structure
        # Assumptions: X.npy and y.csv
        X = np.load(X_path)

        if y_path is not None:
            y = pd.read_csv(y_path).as_matrix()
            y = np.squeeze(y)
        else:
            y = None

        return X, y

    def _prepare_dev_set(self):
        all_ind = np.arange(self.X.shape[0])
        np.random.shuffle(all_ind)
        dev_ind = all_ind[:self.dev_set_size]
        if self.y_path is None:
            return self.X[dev_ind], None
        else:
            return self.X[dev_ind], self.y[dev_ind]  # hope this'll work!
