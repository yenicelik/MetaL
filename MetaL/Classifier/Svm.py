import sklearn.svm

class LinearSVC:

    def __init__(self, X, y):
        # Let cv decide which parameters to take on within this specific model
        # Let test decide what actual score this model has in the very end
        # Unless n_fold cross-validation is used (then we take the average for the test accuracy
        self.X = X
        self.y = y
        self.pred_y = None


        # Hyperparameters
        self.loss = 'squared_hinge'
        self.intercepts = True
        self.penalty = 'l2'
        self.dual = True
        self.tol = 0.0001
        self.C = 1.0
        self.fit_intercept = True
        self.intercept_scaling = 1
        self.class_weight = None
        self.random_state = None
        self.solver = 'liblinear'
        self.max_iter = 1000
        self.multi_class = 'ovr'
        self.verbose = 0
        self.warm_start = False
        self.n_jobs = 1

        self.model = sklearn.svm.LinearSVC(
            loss=self.loss,
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
        )

    def run(self):
        self.model.fit(self.X, self.y)

    def predict(self, X_new):
        out = self.model.predict(X_new)
        return out


class SVC:

    def __init__(self, X, y):
        # Let cv decide which parameters to take on within this specific model
        # Let test decide what actual score this model has in the very end
        # Unless n_fold cross-validation is used (then we take the average for the test accuracy
        self.X = X
        self.y = y
        self.pred_y = None

        # Hyperparameters
        self.loss = 'squared_hinge'
        self.cache_size = 200
        self.kernel = 'rbf'
        self.probability=False
        self.shrinking = True
        self.coef0=0.0
        self.decision_function_shape='ovr'
        self.degree = 3
        self.gamma = 'auto'
        self.intercepts = True
        self.penalty = 'l2'
        self.dual = True
        self.tol = 0.001
        self.C = 1.0
        self.fit_intercept = True
        self.intercept_scaling = 1
        self.class_weight = None
        self.random_state = None
        self.solver = 'liblinear'
        self.max_iter = -1
        self.multi_class = 'ovr'
        self.verbose = 0
        self.warm_start = False
        self.n_jobs = 1

        self.model = sklearn.svm.SVC(
            cache_size=self.cache_size,
            kernel=self.kernel,
            probability=self.probability,
            shrinking=self.shrinking,
            coef0=self.coef0,
            decision_function_shape=self.decision_function_shape,
            degree=self.degree,
            gamma=self.gamma,
            tol=self.tol,
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
        )

    def run(self):
        self.model.fit(self.X, self.y)

    def predict(self, X_new):
        out = self.model.predict(X_new)
        return out


class NuSVC:

    def __init__(self, X, y):
        # Let cv decide which parameters to take on within this specific model
        # Let test decide what actual score this model has in the very end
        # Unless n_fold cross-validation is used (then we take the average for the test accuracy
        self.X = X
        self.y = y
        self.pred_y = None

        # Hyperparameters
        self.nu = 0.5
        self.loss = 'squared_hinge'
        self.cache_size = 200
        self.kernel = 'rbf'
        self.probability=False
        self.shrinking = True
        self.coef0=0.0
        self.decision_function_shape='ovr'
        self.degree = 3
        self.gamma = 'auto'
        self.intercepts = True
        self.penalty = 'l2'
        self.dual = True
        self.tol = 0.001
        self.C = 1.0
        self.fit_intercept = True
        self.intercept_scaling = 1
        self.class_weight = None
        self.random_state = None
        self.solver = 'liblinear'
        self.max_iter = -1
        self.multi_class = 'ovr'
        self.verbose = 0
        self.warm_start = False
        self.n_jobs = 1

        self.model = sklearn.svm.NuSVC(
            nu=self.nu,
            cache_size=self.cache_size,
            kernel=self.kernel,
            probability=self.probability,
            shrinking=self.shrinking,
            coef0=self.coef0,
            decision_function_shape=self.decision_function_shape,
            degree=self.degree,
            gamma=self.gamma,
            tol=self.tol,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            verbose=self.verbose,
        )

    def run(self):
        self.model.fit(self.X, self.y)

    def predict(self, X_new):
        self.pred_y = self.model.predict(X_new)
        return self.pred_y
