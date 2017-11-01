import numpy as np

# Label this class as a 3D volumetric visualizer maybe?
# Assume we have (n, dim1, dim2, dim3) data
# Add a possibility to select a top-left and bottom-right pixel to crop the image by (with clipping values!)


class VolumetricHistogram:
    """ This class viszalizes any 3D object """

    def __init__(self, X_path, y_path=None, descr=""):
        self.RND_SEED = 0  # as we want the experiments to be reproducable!
        print("Starting Volumetric Histogram Preprocessor...")
        X, y = self._import_data(X_path, y_path, descr)
        self.X = self._reshape(X, (-1, 176, 208, 176))
        self.X = X[:, 20:155, 10:195, 25:155]  # Implement an actual 'offset slice' function
        self.descr = descr

        self.step = 5
        self.histo_bins = np.linspace(0, 4500, 11)

    def preprocess(self):
        histos_per_image = (X.shape[1] - step) // step
        histos_per_image *= (X.shape[2] - step) // step
        histos_per_image *= (X.shape[3] - step) // step
  
        histogram = np.zeros((n, histos_per_image, len(histo_bins) - 1))  # (13, 3375, 10) Size
        print(histogram.shape)

        histogram = np.zeros((n, histos_per_image, len(histo_bins) - 1))

        for i in range(self.X.shape[0]):
            c = 0
            for id1 in range(step, X.shape[1], step):
                for id2 in range(step, X.shape[2], step):
                    for id3 in range(step, X.shape[3], step):
                        vol = X_train[
                                i,
                                id1-step:id1,
                                id2-step:id2,
                                id3-step:id3] \
                                .flatten()
                        local_hist = np.histogram(vol, bins=histo_bins)[0]
                        histogram[i, c, :] = local_hist
                        c += 1

        out = np.reshape(
                histogram,
                (X.shape[0], histos_per_image * (len(histo_bins) - 1))
                )

        print("Progress for samples: {:.1f}".format(i/float(X.shape[0])*100))  
        self.save_preprocessed_data(histogram)
   
    def _save_preprocessed_data(self, histogram):
        bin_descr = [(str(x) for x in self.histo_bins)]
        bin_descr = "-".join(bin_descr)
        np.save("VolumetricHistogram" + descr + "_bins-" + bin_descr + str(self.step) + ".npy", histogram)


if __name__ == "__main__":
    Prep = VolumetricHistogram("./X_train.npy", "./y_1.csv")
    Prep = VolumetricHistogram("./X_test.npy", None, "_eval")

    Vis3D.visualize()
    Vis3D_eval.visualize()
