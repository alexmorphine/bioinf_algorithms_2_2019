import numpy as np


class NN:
    """
    Nearest Neighbour base class
    """

    def fit(self, x, y):
        """
        Applying train set
        Args:
            x (pd.DataFrame): dataset with features
            y (pd.DataFrame): set of labels

        """
        self.x = x if type(x) == np.ndarray else x.values
        self.y = y if type(y) == np.ndarray else y.values

    def predict(self, x):
        """
        Function returns predictions for given data
        Args:
            x (pd.DataFrame): input data

        Returns:
            list: predictions
        """

        # output
        y = []

        # if the dataset has more than one row then predict one by one
        if x.ndim > 1:
            for obs in x:
                y.append(self._predict(obs))

        # else just predict for one given row
        else:
            y.append(self._predict(x))
        return y

    def _predict(self, obs):
        """
        Private function that calculates predictions
        Args:
            obs (np.ndarray): input observation

        Returns:
            int: predicted class
        """

        # calculating Euclidean distance
        d = np.sqrt(np.sum((self.x - obs) ** 2, axis=1))

        # retrieving indices of neighbours
        idx = self.neighbours(d)

        # counting each label frequency
        counts = np.bincount(self.y[idx].astype(int))

        # if there is at least one label
        if len(counts):

            # selecting the most frequent label
            prediction = np.argmax(counts)

        # else setting prediction with -1
        else:
            prediction = -1
        return prediction

    def neighbours(self, *args):
        """
        Implemented in child classes
        """
        pass


class KNN(NN):
    """
    Class for KNN algorithm
    """
    name = 'KNN'

    def __init__(self, n):
        """
        Args:
            n (int): number of neighbours to consider
        """
        self.n = int(n)

    def neighbours(self, d):
        """
        Neighbour index finder
        Args:
            d (list): a list of distances

        Returns:
            list: top n closest neighbours indices
        """
        return np.argsort(d)[:self.n]


class RNN(NN):
    """
    Class for RNN algorithm
    """
    name = 'RNN'

    def __init__(self, r):
        """
        Args:
            r (int): radius for neighbours to consider
        """
        self.r = r

    def neighbours(self, d):
        """
        Neighbour index finder
        Args:
            d (list): a list of distances

        Returns:
            list: neighbours in radius indices
        """
        return np.nonzero(d < self.r)[0]


class LOO:
    """
    Class for LOO metric
    """

    def __init__(self, param, nn):
        """
        Args:
            param (float): parameter for NN child class (count or radius)
            nn (cls): class of NN child to be used
        """
        self.param = param
        self.nn = nn

    def calculate(self, x, y):
        """
        Calculates the metric
        Args:
            x (pd.DataFrame): input feature data
            y (pd.DataFrame): input label data

        Returns:
            float: LOO score
        """

        # checking types
        x = x if type(x) == np.ndarray else x.values
        y = y if type(y) == np.ndarray else y.values

        # a list for all scores
        scores = []

        # predicting label for each row
        for i in range(len(x)):
            test_x, test_y = x[i], y[i]
            knn = self.nn(self.param)
            knn.fit(np.delete(x, i, axis=0), np.delete(y, i))
            scores.append(knn.predict(test_x)[0] != test_y)
        return sum(scores) / len(x)

