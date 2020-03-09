import random
from scipy import stats
import numpy as np


def f_entropy(values):
    """
    Calculate information entropy for list of target values (entropy = - sum_i(p_i * log(p_i)))
    Args:
        values (np.ndarray): target values

    Returns:
        float: value of entropy
    """
    # Find each class probability
    p = np.bincount(values) / len(values)

    ep = stats.entropy(p)
    if ep == -float('inf'):
        return 0.0
    return ep


def gini(values):
    """
    Calculate Gini index for list of target values (GI = - sum_i(p_i * (1 - p_i)))
    Args:
        values (np.ndarray): target values

    Returns:
        float: value of GI
    """
    # Find each class probability
    p = np.bincount(values) / len(values)
    return sum(p * (1 - p))


def misclassification(values):
    """
    Calculate Misclassification rate for list of target values (MR = - (1 - max(p_i)))
    Args:
        values (np.ndarray): target values

    Returns:
        float: value of MR
    """
    p = np.bincount(values) / len(values)
    return 1 - max(p)


def information_gain(y, splits, impurity):
    """
    Calculate the information gain for a split of 'y' to 'splits'
    (gain = f(y) - (n_1/n)*f(split_1) - (n_2/n)*f(split_2) ...)
    Args:
        impurity: function to calculate impurity
        y (np.ndarray): array-like initial target values
        splits (list): list of splits, e.g. [[0, 1, 0], [1, 1, 0]]

    Returns:
        float: value of information gain
    """
    splits_entropy = sum([impurity(split) * len(split) for split in splits]) / len(y)
    return impurity(y) - splits_entropy


def split(X, y, threshold):
    """
    Make a binary split of (X, y) using the threshold
    Args:
        X (np.ndarray): feature values for all objects (1 feature)
        y (np.ndarray): target values
        threshold: float threshold for splitting

    Returns:
        list: target values of each split (e.g. [[0, 1, 0], [1, 1, 1]])
    """
    left_mask = (X < threshold)
    right_mask = (X >= threshold)
    return y[left_mask], y[right_mask]


def split_dataset(X, y, column, value):
    """
    Split the dataset (X, y) using X[column] feature at threshold=value
    Args:
        X (np.ndarray): features of objects
        y (np.ndarray): targets
        column (int): index of feature in X
        value (float): value of threshold for X[column]

    Returns:
        list: features and targets of left and right splits
    """
    left_mask, right_mask = get_split_mask(X, column, value)
    left_y = y[left_mask]
    right_y = y[right_mask]
    left_X, right_X = X[left_mask], X[right_mask]
    return left_X, right_X, left_y, right_y


def get_split_mask(X, column, value):
    left_mask = (X[:, column] < value)
    right_mask = (X[:, column] >= value)
    return left_mask, right_mask


class DecisionTree(object):
    """Recursive implementation of decision tree."""

    def __init__(self, criterion_name='entropy'):
        """
        Args:
            criterion_name (str): criterion to use for splitting (default='entropy')
        """
        self.gain = 0
        self.size = 0
        self.column_index = None
        self.threshold = None
        self.outcome = None
        self.outcome_proba = None

        self.left_child = None
        self.right_child = None

        self.criterion_name = criterion_name
        self.unique_targets = None

    def calc_gain(self, y, splits):
        """
        Calculate criterion gain for splits of y
        Args:
            y (np.ndarray): target values
            splits (list): splits, e.g. [[0, 1, 0], [1, 1, 0]]

        Returns:
            float: value of criterion gain
        """
        if self.criterion_name == 'entropy':
            return information_gain(y, splits, f_entropy)
        elif self.criterion_name == 'gini':
            return information_gain(y, splits, gini)
        elif self.criterion_name == 'misclassification':
            return information_gain(y, splits, misclassification)
        else:
            raise NotImplementedError

    @property
    def is_terminal(self):
        """
        Return True if self is leaf else False
        """
        return not bool(self.left_child and self.right_child)

    def _find_splits(self, X):
        """
        Find all possible split values of X
        Args:
            X (np.ndarray): feature vector (X - single feature column)

        Returns:
            list: splitting values
        """
        split_values = set()

        # Get unique values in a sorted order
        x_unique = list(np.unique(X))
        for i in range(1, len(x_unique)):
            # Find a point between two values
            average = (x_unique[i - 1] + x_unique[i]) / 2.
            split_values.add(average)
        return list(split_values)

    def _find_best_split(self, X, y, max_features=None):
        """
        Find best feature and value for a split (greedy algorithm)
        Args:
            X (np.ndarray): features (num_obj x num_features)
            y (np.ndarray): targets
            max_features (int): number of features to use for best split search

        Returns:
            int feature index, float feature threshold value, float criterion gain (for best split)
        """
        # Sample random subset of features
        if max_features is None:
            max_features = X.shape[1]
        subset = random.sample(list(range(0, X.shape[1])), max_features)

        max_gain, max_col, max_val = None, None, None
        for column in subset:
            split_values = self._find_splits(X[:, column])
            for value in split_values:
                splits = split(X[:, column], y, value)
                gain = self.calc_gain(y, splits)
                try:
                    if (max_gain is None) or (gain > max_gain):
                        max_col, max_val, max_gain = column, value, gain
                except:
                    print(max_gain)
                    print(gain)
        return max_col, max_val, max_gain

    def train(self, X, y, unique_targets=None, max_features=None, min_samples_split=2, max_depth=1000, min_gain=0.0001,
              verbose=False):
        """

        Args:
            X (np.ndarray): dataset (num_obj x num_features)
            y (np.ndarray): target
            unique_targets (np.ndarray): unique target values
            max_features (int): int or None, the number of features to consider when looking for the best split
            min_samples_split (int): the minimum number of samples required to split an internal node
            max_depth (int): maximum depth of the tree
            min_gain (float): minimum gain required for splitting
            verbose (bool): print info for debug
        """

        if unique_targets is None:
            unique_targets = sorted(np.unique(y))

        self.size = X.shape[0]
        if max_features is None:
            max_features = X.shape[1]

        try:  # Exit from recursion using assert syntax
            assert (X.shape[0] >= min_samples_split)
            assert (max_depth > 0)

            column, value, gain = self._find_best_split(X, y, max_features)
            self.gain = gain

            if verbose:
                try:
                    print('inputs: {}, max_depth: {}, feature: {}, value: {:.2f}, gain: {:.2f}'
                          .format(self.size, max_depth, column, value, gain))
                except TypeError:
                    print(f'inputs: {self.size}, max_depth: {max_depth}')

            assert gain is not None
            assert (gain > min_gain)

            self.column_index = column
            self.threshold = value

            # Split dataset
            left_X, right_X, left_y, right_y = split_dataset(X, y, column, value)

            # Grow left and right child
            self.left_child = DecisionTree(self.criterion_name)
            self.left_child.train(left_X, left_y, unique_targets, max_features, min_samples_split, max_depth - 1,
                                  min_gain, verbose)

            self.right_child = DecisionTree(self.criterion_name)
            self.right_child.train(right_X, right_y, unique_targets, max_features, min_samples_split, max_depth - 1,
                                   min_gain, verbose)

        except AssertionError:
            self._calculate_leaf_value(y, unique_targets)
            self.left_child = None
            self.right_child = None
            self.depth = max_depth

    def _calculate_leaf_value(self, y, unique_targets):
        """
        Find output value for leaf and store it in self.outcome & self.outcome_proba
        Args:
            y (dict): {'y': array-like of targets}
            unique_targets (np.ndarray): unique target values
        """
        # Most probable class for classification task
        uniques, counts = np.unique(y, return_counts=True)
        self.outcome = uniques.astype(np.int)[np.argmax(counts)]

        # Outcome probabilities
        counts_normed = counts / counts.sum()
        probs = {label: prob for label, prob in zip(uniques, counts_normed)}

        self.outcome_proba = []
        for unique_target in unique_targets:
            p = probs[unique_target] if unique_target in probs else 0
            self.outcome_proba.append(p)
        self.outcome_proba = np.asarray(self.outcome_proba)

    def predict_row(self, row):
        """
        Predict for single row
        Args:
            row (np.ndarray): row for which to predict

        Returns:
            int: predicted result
        """

        if not self.is_terminal:
            if row[self.column_index] < self.threshold:
                return self.left_child.predict_row(row)
            else:
                return self.right_child.predict_row(row)
        return self.outcome

    def predict_proba_row(self, row):
        """
        Predict probability for single row
        Args:
            row (np.ndarray): row for which to predict

        Returns:
            list: predicted probabilities for each class
        """
        if not self.is_terminal:
            if row[self.column_index] < self.threshold:
                return self.left_child.predict_proba_row(row)
            else:
                return self.right_child.predict_proba_row(row)
        return self.outcome_proba

    def predict(self, X):
        """
        Predict for X
        Args:
            X (np.ndarray): feature data

        Returns:
            list: predicted classes
        """

        result = np.zeros(shape=(X.shape[0]), dtype=np.int)
        for i in range(X.shape[0]):
            result[i] = self.predict_row(X[i, :])
        return result

    def predict_proba(self, X):
        """
        Predict probabilities for X
        Args:
            X (np.ndarray): feature data

        Returns:
            list: predicted probabilities for each class for each row
        """
        result = np.zeros(shape=(X.shape[0], 2))
        for i in range(X.shape[0]):
            result[i] = self.predict_proba_row(X[i, :])
        return result