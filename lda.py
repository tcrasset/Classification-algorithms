"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin



class LinearDiscriminantAnalysis(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        """Fit a linear discriminant analysis model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        # Stores for each class k a tuple 
        # (nb_occurance of class k, sum of attributes of all samples belonging to k)
        # dictionnary[k][0] =  nb_occurance
        # dictionnary[k][1] =  sum of attributes of all samples belonging to k
        dictionary = {}

        for _X, _y in zip(X,y):
            if _y in dictionary:
                dictionary[_y][0] += 1
                dictionary[_y][1] = np.add(dictionary[_y][1],_X)
            else:
                dictionary[_y][0] == 1
                dictionary[_y][1] = np.array(_X)

        for key, value in zip(dictionary.keys(), dictionary.values()):
            pi = np.append(key, value[0]/y.shape[0]) # Probability of belonging to class
            mu = np.append(key, value[1]/value[0]) # Mean of all attributes

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # ====================
        # TODO your code here.
        # ====================

        pass

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # ====================
        # TODO your code here.
        # ====================

        pass

if __name__ == "__main__":
    from data import make_data
    from plot import plot_boundary
