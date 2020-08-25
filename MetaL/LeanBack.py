from Importer.Importer import Importer
from Importer.PreprocessedImporter import PreprocessedImporter
from Visualizer.Visualizer3D import Visualizer3D

from Preprocess.VolumetricHistogram import VolumetricHistogram
from Preprocess.Identity import Identity

from Classifier.Random import Random
from Classifier.LogisticRegression import LogisticRegression
from Classifier.Svm import LinearSVC, SVC, NuSVC

from Losses.Losses import Losses

DEV = True


class LeanBack:

    def __init__(self):

        # Single thread. Import the data into the program
        ImporterTrain = Importer("./X_train.npy", "./y_1.csv", DEV)
        ImporterEval = Importer("./X_test.npy", None, DEV)

        X = ImporterTrain.getX()
        y = ImporterTrain.gety()
        X_eval = ImporterEval.getX()

        # Single thread. Visualize the data
        Vis3D = Visualizer3D(X, y, "X")
        Vis3DEval = Visualizer3D(X_eval, None, "X_eval")
        Vis3D.visualize()
        Vis3DEval.visualize()

        # Signle thread, for the simple reason that we cannot bloat up the memory. Shouldn't be a bottleneck though
        Preprocessors = [Identity, VolumetricHistogram]
        for Preprocessor in Preprocessors:
            InstancePreprocessor = Preprocessor(X, y, "X")
            InstancePreprocessor.preprocess()
            X, y = InstancePreprocessor.get_X_y()

            print("Data we will be working on has shape: ")
            print("X: ", X.shape)
            print("y: ", y.shape)

            # Multiple threads. Create multiple threads that concurrently fit a model and predict using that value
            Classifiers = [Random, LogisticRegression, LinearSVC, SVC, NuSVC]
            Loss = Losses()
            # -------- START HERE
            for Classifier in Classifiers:
                print("Starting classifier: ", str(Classifier))
                InstanceClassifier = Classifier(X, y)
                InstanceClassifier.run()

                pred_y = InstanceClassifier.predict(X)
                Loss.add_data(y, pred_y),
                loss = Loss.mean_squared_error()
                print("MSE: {:.3f}".format(float(loss)))
            # -------- JOIN HERE
   
if __name__ == "__main__":
    lb = LeanBack()
