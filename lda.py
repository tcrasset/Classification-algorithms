"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
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

        # Probability of belonging to given class
        # Shape :
        # self.pi_ = {class1 : probability,
        #       class2 : probability,
        #        ...
        #      }
        self.pi_ = {}

        # Elementwise mean of attributes per class 
        # Shape :
        # self.mu_ = {class1 : [mean of attributes],
        #       class2 : [mean of attributes],
        #        ...
        #      }
        self.mu_ = {}
        
        # Covariance matrix
        self.Sigma_ = None

        # Stores for each class k a tuple 
        # (nb_occurance of class k, list of all samples belonging to k)
        # Shape :
        # dictionnary = {class1 : (nb_occurance, list of samples),
        #                class2 : (nb_occurance, list of samples),
        #                ...
        #               }
        dictionary = {}


        #Separating the points into their class
        for _X, _y in zip(X,y):
            if _y in dictionary:
                dictionary[_y][0] += 1 
                dictionary[_y][1].append(_X)
            else:
                dictionary[_y] = [1, [_X] ]

        # Getting all the unique classes
        self.classes_ = dictionary.keys()
        
        #Computing self.pi_ and self.mu_
        X_normlist = []
        for key, value in zip(dictionary.keys(), dictionary.values()):
            attributes_array = np.array(value[1])
            self.mu_[key] = np.mean(attributes_array,axis=0) # Elementwise mean of attributes per class
            self.pi_[key] = value[0]/y.shape[0] # Probability of belonging to class
            #Normalize attributes with the mean of their class
            X_normlist.append(attributes_array - self.mu_[key])

        #Concatenating all the points into one array again
        X_norm = np.concatenate([xnorm for xnorm in X_normlist], axis=0) 

        #Covariance matrix
        self.Sigma_ = np.cov(X_norm, rowvar=False)

        return self

    def _classDensityFunction(self, x, mu_k):
        """ 
        Computes the class density function
        """
        constant = 1/(2*math.pi**(x.shape[0]/2)*np.sqrt(np.linalg.det(self.Sigma_)))
        xnorm = x - mu_k
        exposant = math.exp(-0.5 * np.transpose(xnorm)@np.linalg.inv(self.Sigma_)@xnorm)
        return constant*exposant

    def _probpost(self, x, k):
        """ 
        Computes the probability a posteriori of x belonging
        to class k
        """
        summation = 0
        for key in self.mu_.keys():
            summation += self._classDensityFunction(x, self.mu_[key]) * self.pi_[key]
        return self._classDensityFunction(x, self.mu_[k]) * self.pi_[k] / summation

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

        try:
            getattr(self, "Sigma_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        y = np.empty(X.shape[0])
        indices = np.empty(X.shape[0],dtype=np.int64)
        p = self.predict_proba(X)
        np.argmax(p, axis=1, out=indices)
        classes = sorted(self.classes_)

        y = [classes[i] for i in indices]
        return y

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

        #Computing probability a posteriori of X belonging to a certain class
        classes = sorted(self.classes_)
        p = []
        for x in X:
            for k in classes:
                p.append(self._probpost(x, k))

        p = np.array(p).reshape((X.shape[0],len(classes)))
        return p

if __name__ == "__main__":
    from data import make_dataset1
    from plot import plot_boundary
    X, y = make_dataset1(1500,666)
    estimator = LinearDiscriminantAnalysis().fit(X, y)

    y_pred = estimator.predict(X)
    print(y)
