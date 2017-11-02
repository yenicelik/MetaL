from Importer.Importer import Importer
from Importer.PreprocessedImporter import PreprocessedImporter
from Visualizer.Visualizer3D import Visualizer3D
from Preprocess.VolumetricHistogram import VolumetricHistogram

from Classifier.Random import Random
from Classifier.LogisticRegression import LogisticRegression

from Losses.Losses import Losses

DEV = True


class LeanBack:

    def __init__(self):

        ImporterTrain = Importer("./X_train.npy", "./y_1.csv", DEV)
        ImporterEval = Importer("./X_test.npy", None, DEV)
        X = ImporterTrain.getX()
        y = ImporterTrain.gety()
        X_eval = ImporterEval.getX()

        Vis3D = Visualizer3D(X, y, "X")
        Vis3DEval = Visualizer3D(X_eval, None, "X_eval")

        Vis3D.visualize()
        Vis3DEval.visualize()

        Prep = VolumetricHistogram(X, y, "X")
        Prep.preprocess()
        PrepEval = VolumetricHistogram(X_eval, None, "X_eval")
        PrepEval.preprocess()

        # Collect all classifiers here
        Classifiers = [Random, LogisticRegression]

        PrepImporter = PreprocessedImporter("./X_train.npy", "./y_1.csv", DEV)
        Loss = Losses()

        while not PrepImporter.done:
            X, y = PrepImporter.get_preprocessed_X_y()

            for Classifier in Classifiers:
                print("Starting classifier: ", str(Classifier))
                InstanceClassifier = Classifier(X, y)
                InstanceClassifier.run()
                pred_y = InstanceClassifier.predict(X)
                print("Pred_y: ", pred_y)
                Loss.add_data(y, pred_y)
                loss = Loss.mean_squared_error()
                print("MSE: {:.3f}".format(float(loss)))


if __name__ == "__main__":
    lb = LeanBack()
