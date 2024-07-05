from __future__ import annotations
from typing import Tuple, NoReturn
from base_estimator import BaseEstimator
import numpy as np
from itertools import product
from loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_error= np.inf
        for j, sign in product(range(X.shape[1]), [-1, 1]):
            # Find the best threshold for the given feature and sign
            thr, thr_err = self._find_threshold(X[:, j], y, sign)

            if thr_err < min_error:
                # Update the best threshold, feature and sign
                self.threshold_, self.j_, self.sign_, min_error = thr, j, sign, thr_err


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        n_samples= X.shape[0]
        predictions= np.ones(n_samples)
        predictions[X[:, self.j_] < self.threshold_]= -self.sign_
        predictions[X[:, self.j_] >= self.threshold_]= self.sign_

        return predictions


    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        ids = np.argsort(values)
        values, labels = values[ids], labels[ids]

        # Loss for classifying all as `sign` - namely, if threshold is smaller than values[0]
        loss = np.sum(np.abs(labels)[np.sign(labels) == sign])

        # Loss of classifying threshold being each of the values given
        loss = np.append(loss, loss - np.cumsum(labels * sign))

        id = np.argmin(loss)
        return np.concatenate([[-np.inf], values[1:], [np.inf]])[id], loss[id]


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        predictions= self._predict(X)
        loss= misclassification_error(y, predictions)
        return loss
