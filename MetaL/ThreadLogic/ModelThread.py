from .Importer.Importer import Importer
import multiprocessing

class ModelThread(multiprocessing.Process):

    def __init__(self, Model, X, y=None):
        # Objects
        self.Model = Model
        self.X = X
        self.y = y
        self.pred_y = None
        self.done = False

    def run(self):
        self.Model.fit(X, y)
        self.pred_y = self.Model.predict(X))
        self.done = True

        # Possibly do a Hyperparameter Grid search. All possible values should be saved within the respective model class for that

    def __str__(self):
        print(
            "PreprocessorThread: { \n" +
            "\tPreprocessor: " + str(self.Preprocessor) + ",\n" +
            "\tX_path: " + str(X_path) + ",\n" + 
            "\ty_path: " + str(y_path) + ",\n" +
            "}"
        )
