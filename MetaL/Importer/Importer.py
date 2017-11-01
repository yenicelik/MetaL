import pandas as pd
import numpy as np


class Importer:
    """ This class viszalizes any 3D object """

    def __init__(self, X_path, y_path=None, dev=False):
        self.RND_SEED = 0  # as we want the experiments to be reproducable!
        if y_path is None:
            self.X, _ = self._import_data(X_path, y_path)
        else:
            self.X, self.y = self._import_data(X_path, y_path)
        self.dev = dev

        # Hyperparameters
        self.max_train_ind = int(0.7 * self.X.shape[0])
        self.max_cv_ind = int(0.15 * self.X.shape[0])
        self.max_test_ind = int(0.15 * self.X.shape[0])
        self.dev_set_size = int(self.X.shape[0]**0.8)

        if dev:
            self.X, self.y = self._prepare_dev_set()  # is X and y is returned always applicable

    def _import_data(self, X_path, y_path=None):
        # Currently assume that we have pandas-structure
        # Assumptions: X.npy and y.csv
        X = np.load(X_path)

        print("General dimensions: ")
        print("X: ", X.shape)

        if y_path is not None:
            y = pd.read_csv(y_path).as_matrix()
            print("y: ", y.shape)
        else:
            y = None

        return X, y

    def _prepare_dev_set(self):
        all_ind = np.arange(self.X.shape[0])
        np.random.shuffle(all_ind)
        dev_ind = all_ind[:self.dev_set_size]
        print(dev_ind)
        if self.dev:
            return self.X[dev_ind], None
        else:
            return self.X[dev_ind], self.y[dev_ind]  # hope this'll work!


if __name__ == "__main__":
    Prep = Importer("./X_train.npy", "./y_1.csv", True)
    Prep = Importer("./X_test.npy", None, "_eval")
