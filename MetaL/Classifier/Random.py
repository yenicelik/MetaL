import numpy as np

class Random:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.pred_y = None

    def run(self):
        rnd_ind = np.arange(len(self.y))
        np.random.shuffle(rnd_ind)
        self.pred_y = self.y[rnd_ind]

    def predict(self, X_new):
        self.run()
        return self.pred_y
