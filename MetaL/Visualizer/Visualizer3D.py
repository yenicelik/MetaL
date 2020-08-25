import sys
import numpy as np
import matplotlib.pyplot as plt

# Label this class as a 3D volumetric visualizer maybe?
# Assume we have (n, dim1, dim2, dim3) data
# Add a possibility to crop the image by (with clipping values!)


class Visualizer3D:
    """ This class viszalizes any 3D object """

    def __init__(self, X, y=None, descr=""):
        self.RND_SEED = 0  # as we want the experiments to be reproducable!
        print("Starting visualizer...")
        X = self._reshape(X, (-1, 176, 208, 176))
        self.X = X
        self.y = y
        self.descr = descr

    def visualize(self):
        """ Visualize the 3D data from all possible angles """
        for i in range(1, 4):
            self._show_image_by_dim(self.X, i, self.descr)

    def _reshape(self, X, dim):
        # We have to reshape if we have a row-only dataset
        return np.reshape(X, dim)

    def _show_image_by_dim(self, X, dim, descr=""):
        # Print the middle-slice from the image
        np.random.seed(self.RND_SEED)
        training_sample = np.random.randint(0, X.shape[0])
        s1 = X.shape[1] // 2
        s2 = X.shape[2] // 2
        s3 = X.shape[3] // 2

        if dim == 1:
            img_slice = np.squeeze(X[training_sample, s1, :, :])
        elif dim == 2:
            img_slice = np.squeeze(X[training_sample, :, s2, :])
        elif dim == 3:
            img_slice = np.squeeze(X[training_sample, :, :, s3])
        else:
            print("No real dimension was given! dim must be from amongst {1, 2, 3}")
            print("From _show_image_by_dim")
            sys.exit(16)

        plt.imshow(img_slice)
        print("Saving Visualized data: (dim) (descr)", dim, descr)
        plt.savefig(descr + "_dim" + str(dim) + descr)

    def _slice_dimension(self, X, dim, off_min, off_max):
        # From the respective dimensions, return only what is from off_min, up to off_max
        pass

if __name__ == "__main__":
    Vis3D = Visualizer3D("./X_train.npy", "./y_1.csv")
    Vis3D_eval = Visualizer3D("./X_test.npy", None, "_eval")

    Vis3D.visualize()
    Vis3D_eval.visualize()
