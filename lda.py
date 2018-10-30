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

        #Probability of belonging to given class

        # Shape :
        # pi = {class1 : probability,
        #       class2 : probability,
        #        ...
        #      }
        pi = {}

        # Elementwise mean of attributes per class 
        # Shape :
        # mu = {class1 : [mean of attributes],
        #       class2 : [mean of attributes],
        #        ...
        #      }
        mu = {}

        X_norm = []
        Sigma = None

        dictionary = {}
        # Stores for each class k a tuple 
        # (nb_occurance of class k, list of all samples belonging to k)
        # dictionnary[k][0] =  nb_occurance
        # dictionnary[k][1] =  list of all samples belonging to k
        dictionary = {}

        #Separating the points into their class
        for _X, _y in zip(X,y):
            if _y in dictionary:
                dictionary[_y][0] += 1 
                dictionary[_y][1].append(_X)
            else:
                dictionary[_y] = [1, [_X] ]

        #Computing pi and mu
        for key, value in zip(dictionary.keys(), dictionary.values()):
            attributes_array = np.array(value[1])
            mu[key] = np.mean(attributes_array,axis=0) # Elementwise mean of attributes per class
            pi[key] = value[0]/y.shape[0] # Probability of belonging to class
            #Normalize attributes with the mean of their class
            X_norm.append(attributes_array - mu[key])

        #Concatenating all the points into one array again
        X_norm[0] = np.array(X_norm[0])
        X_norm[1] = np.array(X_norm[1])
        X_norm = np.concatenate((X_norm[0], X_norm[1]), axis=0)

        #Covariance matrix
        Sigma = np.cov(X_norm, rowvar=False)

        const = 1/(2*math.pi**(X.shape[1]/2)*np.sqrt(np.abs(Sigma)))
        for _x, _y in zip(X,y):
            _xnorm = _x - mu[_y]
            exposant = math.exp(-0.5 * np.transpose(_xnorm)@np.linalg.inv(Sigma)@_xnorm)
            function = const*exposant
            
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
    from data import make_dataset1
    from plot import plot_boundary
