import os
import numpy as np

# Label this class as a 3D volumetric visualizer maybe?
# Assume we have (n, dim1, dim2, dim3) data
# Add a possibility to select a top-left and bottom-right pixel to crop the image by (with clipping values!)

class Identity:
    """ This class viszalizes any 3D object """

    def __init__(self, X, y=None, descr=""):
        self.RND_SEED = 0  # as we want the experiments to be reproducable!
        print("Starting Identity Preprocessor...")
        self.X = X
        self.y = y
        self.descr = descr

    def preprocess(self):
        self._save_preprocessed_data(self.X)
        return self.X

    def get_X_y(self):
        return self.X, self.y

    def _save_preprocessed_data(self, histogram):
        directory = "./tmp_data/Identity/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(
            directory + self.descr + ".npy",
            histogram
                )
