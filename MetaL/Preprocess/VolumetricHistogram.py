import os
import numpy as np

# Label this class as a 3D volumetric visualizer maybe?
# Assume we have (n, dim1, dim2, dim3) data
# Add a possibility to select a top-left and bottom-right pixel to crop the image by (with clipping values!)


class VolumetricHistogram:
    """ This class viszalizes any 3D object """

    def __init__(self, X, y=None, descr=""):
        self.RND_SEED = 0  # as we want the experiments to be reproducable!
        print("Starting Volumetric Histogram Preprocessor...")
        self.X = self._reshape(X, (-1, 176, 208, 176))
        self.X = self.X[:, 20:155, 10:195, 25:155]  # Implement an actual 'offset slice' function
        self.y = y
        self.descr = descr

        self.step = 5
        self.histo_bins = np.linspace(0, 4500, 11)

    def preprocess(self):
        histos_per_image = (self.X.shape[1] - self.step) // self.step
        histos_per_image *= (self.X.shape[2] - self.step) // self.step
        histos_per_image *= (self.X.shape[3] - self.step) // self.step
        histogram = np.zeros((self.X.shape[0], histos_per_image, len(self.histo_bins) - 1))

        for i in range(self.X.shape[0]):
            c = 0
            for id1 in range(self.step, self.X.shape[1], self.step):
                for id2 in range(self.step, self.X.shape[2], self.step):
                    for id3 in range(self.step, self.X.shape[3], self.step):
                        vol = self.X[
                                i,
                                id1-self.step:id1,
                                id2-self.step:id2,
                                id3-self.step:id3] \
                                .flatten()
                        local_hist = np.histogram(vol, bins=self.histo_bins)[0]
                        histogram[i, c, :] = local_hist
                        c += 1

            print("Progress for samples: {:.1f}".format(i/float(self.X.shape[0])*100))

        out = np.reshape(
                histogram,
                (self.X.shape[0], histos_per_image * (len(self.histo_bins) - 1))
                )

        self._save_preprocessed_data(out)

        self.X = out

        return out

    def get_X_y(self):
        return self.X, self.y

    def _save_preprocessed_data(self, histogram):
        bin_descr = [str(x) for x in self.histo_bins]
        bin_descr = "-".join(bin_descr)
        directory = "./tmp_data/VolumetricHistogram/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(
            directory + self.descr + "_bins-" + bin_descr + "_step-" + str(self.step) + ".npy",
            histogram
                )

    def _reshape(self, X, dim):
        # We have to reshape if we have a row-only dataset
        return np.reshape(X, dim)

