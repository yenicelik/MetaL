import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Label this class as a 3D volumetric visualizer maybe?
#Assume we have (n, dim1, dim2, dim3) data
#Add a possibility to select a top-left and bottom-right pixel to crop the image by (with clipping values!)

RND_SEED = 0 #as we want the experiments to be reproducable!

def import_data(X_path, y_path=None, descr=""):
    #Currently assume that we have pandas-structure
    #Assumptions: X.npy and y.csv
    X = np.load(X_path)

    print("General dimensions: ")
    print("X" + descr + ": ", X.shape)

    if y_path is not None:
        y = pd.read_csv(y_path).as_matrix()
        print("y" + descr + ": ", y.shape)
    else:
        y = None

    return X, y


def reshape_to_dim(X, dim):
    # We have to reshape if we have a row-only dataset
    return np.reshape(X, dim)


def print_dim_image(X, dim, descr=""):
    # Print the middle-slice from the image
    np.random.seed(RND_SEED)
    training_sample = np.random.randint(0, X.shape[0])
    s1 = X.shape[1] // 2
    s2 = X.shape[2] // 2
    s3 = X.shape[3] // 2

    if dim == 1:
        img_slice = np.squeeze(X[training_sample, s1, :, :])
    elif dim == 2:
        img_slice = np.squeeze(X[training_sample, :, s2, : ])
    elif dim == 3:
        img_slice = np.squeeze(X[training_sample, :, :, s3])
    else: 
        print("No real dimension was given! dim must be from amongst {1, 2, 3}")
        print("From print_dim_image")
        sys.exit(16)

    img = plt.imshow(img_slice)
    plt.savefig(descr + ":dim" + str(dim))
#    plt.show()


def slice_dimension(X, dim, off_min, off_max):
    # From the respective dimensions, return only what is from off_min, up to off_max
    pass

if __name__ == "__main__":
    print("Starting visualizer...")

    X, y = import_data("./X_train.npy", "./y_1.csv")
    X_eval = import_data("./X_test.npy", None, "_eval")

    X = reshape_to_dim(X, (-1, 176, 208, 176))
    for i in range(1, 4):
        print_dim_image(X, i)






