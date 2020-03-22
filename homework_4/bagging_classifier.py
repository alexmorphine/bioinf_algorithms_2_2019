import numpy as np


class BaggingClassifier:

    def __init__(self, base_model_class, n_base_models, **base_model_params):
        """

        Args:
            base_model_class: class of a model to be used as base model
            n_base_models (int):  number of base models to use in bagging
            **base_model_params: params to pass to base_models
        """
        self.n_base_models = n_base_models
        self.base_models = [self._create_model_(base_model_class, **base_model_params)
                            for _ in range(self.n_base_models)]

    def _create_model_(self, base_model_class, **base_model_params):
        """
        Initializing a base model
        Args:
            base_model_class: class of a model to be used as base model
            **base_model_params: params to pass to base_models

        Returns:

        """
        return base_model_class(**base_model_params)

    def _create_bagging_subsample_(self, X, y):
        """
        Create subsample of X using bagging sampling
        Args:
            X (np.array): features
            y (np.array): targets

        Returns:
            X_bagging (np.array): features of subsample
            y_bagging (np.array): targets of subsample
        """
        bagging_indexes = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        X_bagging = X[bagging_indexes]
        y_bagging = y[bagging_indexes]
        return X_bagging, y_bagging

    def train(self, X, y):
        """
        Train method of the classifier
        Args:
            X (np.array): features
            y (np.array): targets
        """
        for base_model in self.base_models:
            X_bagging, y_bagging = self._create_bagging_subsample_(X, y)
            base_model.train(X_bagging, y_bagging)

    def predict(self, X):
        """
        Make prediction for X
        Args:
            X (np.array): features

        Returns:
            target predictions
        """
        base_models_predictions = np.asarray([base_model.predict(X) for base_model in self.base_models],
                                             dtype=np.int).transpose(1, 0)
        result = []
        for base_model_prediction in base_models_predictions:
            uniques, counts = np.unique(base_model_prediction, return_counts=True)
            result.append(uniques[np.argmax(counts)])
        return np.asarray(result)

    def predict_proba(self, X):
        """
        Make probability prediction for X
        Args:
            X (np.array): features

        Returns:
            target probabilities predictions
        """
        base_models_predictions = np.asarray([base_model.predict_proba(X)
                                              for base_model in self.base_models], dtype=np.float).transpose(1, 0, 2)
        return base_models_predictions.mean(axis=1)
