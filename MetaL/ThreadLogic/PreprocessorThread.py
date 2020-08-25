from .Importer.Importer import Importer
import multiprocessing

class PreprocessorThread(multiprocessing.Process):

    def __init__(self, Preprocessor, X, y=None):
        # Objects
        self.Preprocessor = Preprocessor
        self.done = False

    def run(self):
        Preprocessor.preprocess()
        self.done = True
        
    def __str__(self):
        print(
            "PreprocessorThread: { \n" +
            "\tPreprocessor: " + str(self.Preprocessor) + ",\n" +
            "\tX_path: " + str(X_path) + ",\n" + 
            "\ty_path: " + str(y_path) + ",\n" +
            "}"
        )
