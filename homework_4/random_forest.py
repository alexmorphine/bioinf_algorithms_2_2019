from bagging_classifier import BaggingClassifier


class RandomForestClassifier(BaggingClassifier):
    """ Random forest classifier class """

    def __init__(self, base_model_class, n_base_models, n_features=None, **base_model_params):
        """
        Args:
            base_model_class: base model class object
            n_base_models (int): number of base models to use
            n_features (int): number of features to use in each model
            **base_model_params: parameters for base model class
        """
        self.n_base_models = n_base_models
        self.n_features = n_features
        self.base_models = [self._create_model_(base_model_class, **base_model_params)
                            for _ in range(self.n_base_models)]

    def train(self, X, y):
        """
        Train method of the classifier
        Args:
            X: train X
            y: target
        """

        for base_model in self.base_models:
            X_bagging, y_bagging = self._create_bagging_subsample_(X, y)
            base_model.train(X_bagging, y_bagging, max_features=X.shape[1] - self.n_features)