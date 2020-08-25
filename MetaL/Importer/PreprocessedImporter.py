import os
import pandas as pd
import numpy as np


class PreprocessedImporter:
    """ This class viszalizes any 3D object """

    def __init__(self, X_path, y_path=None, dev=False):
        self.RND_SEED = 0  # as we want the experiments to be reproducable!

        f = []
        for (dirpath, dirnames, filenames) in os.walk("./tmp_data/"):
            f.extend(dirnames)
        self.files = []
        for dire in f:
            for (dirpath, dirnames, filenames) in os.walk("./tmp_data/" + dire):
                for ele in filenames:
                    if ele[0] != ".":
                        self.files.append(("./tmp_data/" + dire + "/" + ele, dire))

        if y_path is None:
            self.X, _ = self._import_data(X_path, y_path)
            self.y = None
        else:
            self.X, self.y = self._import_data(X_path, y_path)

        self.X_path = X_path
        self.y_path = y_path
        self.dev = dev
        self.counter = 0
        self.max_counter = len(self.files) - 1
        self.done = False

        # Hyperparameters
        self.dev_set_size = 2  # int(self.X.shape[0]*0.1) # **0.7 / **0.8 good values

    def get_preprocessed_X_y(self):
        filename = self.files[self.counter][0]
        filedescr = self.files[self.counter][1]
        self.X, self.y, = self._import_data(filename, self.y_path)

        self.counter += 1
        if self.counter >= self.max_counter:
            self.done = True

        if self.y_path is None:
            self.X, _ = self._prepare_dev_set()
            self.y = None
        else:
            self.X, self.y = self._prepare_dev_set()

        return self.X, self.y

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

if __name__ == "__main__":
    Prep = PreprocessedImporter("./X_train.npy", "./y_1.csv", True)
    Prep = PreprocessedImporter("./X_test.npy", None, "_eval")
