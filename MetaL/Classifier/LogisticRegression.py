from sklearn import linear_model


class LogisticRegression:

    def __init__(self, X, y):
        # Let cv decide which parameters to take on within this specific model
        # Let test decide what actual score this model has in the very end
        # Unless n_fold cross-validation is used (then we take the average for the test accuracy
        self.X = X
        self.y = y
        self.pred_y = None

        # Hyperparameters
        self.intercepts = True
        self.penalty = 'l2'
        self.dual = False
        self.tol = 0.0001
        self.C = 1.0
        self.fit_intercept = True
        self.intercept_scaling = 1
        self.class_weight = None
        self.random_state = None
        self.solver = 'liblinear'
        self.max_iter = 100
        self.multi_class = 'ovr'
        self.verbose = 0
        self.warm_start = False
        self.n_jobs = 1

        self.model = linear_model.LogisticRegression(
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
            verbose=self.verbose,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs
        )

    def run(self):
        self.model.fit(self.X, self.y)

    def predict(self, X_new):
        out = self.model.predict(X_new)
        return out
