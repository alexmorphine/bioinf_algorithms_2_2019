import numpy as np
from sklearn.metrics import accuracy_score


class PocketPerceptron:
    """
    Pocket algorithm for a perceptron
    """
    def __init__(self, max_iterations=1000, early_stop=False, tolerance_steps=5, tolerance_difference=10e-6):
        """
        Args:
            max_iterations (int): maximum number of iterations
            early_stop (bool): use early stopping for training
            tolerance_steps (int): number of steps accuracy shouldn't change to stop training
            tolerance_difference (float): minimum change between accuracies of sequential steps
        """
        self.early_stop = early_stop
        self.max_iterations = max_iterations
        self.accuracies = []
        self.iterations = 0
        self.tolerance_steps = tolerance_steps
        self.tolerance_difference = tolerance_difference

    def fit(self, x, y):
        """
        Train function for the classifier
        Args:
            x (np.array): features
            y (np.array): target
        """
        b = np.ones((len(x), x.shape[1] + 1))
        b[:, 1:] = x

        # initializing weights
        self.weights = np.random.rand(b.shape[1])
        weights_used = []

        predictions = np.sign(np.dot(b, self.weights))
        while self.iterations < self.max_iterations:
            incorrect = np.argwhere(predictions != y).flatten()

            accuracy = accuracy_score(predictions, y)

            self.accuracies.append(accuracy)
            self.weights += np.dot(b[incorrect].T, y[incorrect])
            weights_used.append(self.weights)

            predictions = np.sign(np.dot(b, self.weights))

            # early stopping
            if self.early_stop:
                if len(self.accuracies) >= self.tolerance_steps and \
                        (np.array_equal(self.accuracies[:-self.tolerance_steps],
                                        self.accuracies[:-self.tolerance_steps:-1]) or
                         np.abs(accuracy - self.accuracies[-2]) < self.tolerance_difference):
                    break
            self.iterations += 1

            # best weights
            self.weights = weights_used[np.argmax(self.accuracies)]

    def predict(self, x):
        """
        Predict for the classifier
        Args:
            x (np.array): features

        Returns:
            predictions (np.array)
        """
        b = np.ones((len(x), x.shape[1] + 1))
        b[:, 1:] = x
        return np.sign(np.dot(b, self.weights))
