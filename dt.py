"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from data import make_dataset1, make_dataset2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from plot import plot_boundary



def compute_accuracy(nb_gen, max_depth, nb_points):
    """Computes the test set accurencies over n generations of the dataset
    using the DecisionTreeClassifier class from sklearn.tree with a
    particular max depth.

    Parameters
    ----------
    -   nb_gen : number of generations of the dataset.
    -   max_depth : maximum depth of the decision tree for the DT model.
    -   nb_points : number of samples.

    Returns
    -------
    accuracy : a list of the test set accuracies of the different
    generations.
    """
    accuracy = []

    for generation in range(nb_gen):

        X, y = make_dataset2(nb_points, generation)
        X_ls, X_ts, y_ls, y_ts = train_test_split(X, y, train_size=.8)
        
        if max_depth == "None":
            estimator = DecisionTreeClassifier().fit(X_ls, y_ls)
        else:
            estimator = DecisionTreeClassifier(max_depth=max_depth).fit(X_ls, y_ls)

        y_pred = estimator.predict(X_ts)
        accuracy.append(accuracy_score(y_ts, y_pred))

        if generation == 1:
            plot_boundary("DT max_depth {}".format(max_depth), estimator, X_ts, y_ts, 0.1)

    return accuracy

if __name__ == "__main__":
    nb_gen = 5
    nb_points = 1500
    max_depths = [1, 2, 4, 8, "None"]
    
    for max_depth in max_depths:
        print("Maximal depth : {}".format(max_depth))
        print("Mean \t STD")
        accuracy = np.array(compute_accuracy(nb_gen, max_depth, nb_points))
        print("{:.3f} \t {:.4f}".format(accuracy.mean(), accuracy.std()))
