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
# (Question 1)

# Put your funtions here
# ...

def trainEstimator(nbGen, maxdepth):
    accuracy_arr = []
    for generation in range(nbGen):
        nbPoints = 1500
        seed = generation
        X, y = make_dataset2(nbPoints, seed)

        X_ls, X_ts, y_ls, y_ts = train_test_split(X, y, train_size = 1200, test_size = 300)
        
        if maxdepth == "None":
            estimator = DecisionTreeClassifier().fit(X_ls,y_ls)
        else:
            estimator = DecisionTreeClassifier(max_depth=maxdepth).fit(X_ls,y_ls)

        y_pred = estimator.predict(X_ts)
        accuracy_arr.append(accuracy_score(y_ts, y_pred))

        if generation == 1:
            plot_boundary("DT maxdepth {}".format(maxdepth),estimator , X_ts, y_ts, 0.1)
    return accuracy_arr


if __name__ == "__main__":
    nbGen = 5
    maxdepths = [1, 2, 4, 8, "None"]
    
    for maxdepth in maxdepths:
        print("Maximal depth : {}".format(maxdepth))
        print("Mean \t STD")
        accuracy = np.array(trainEstimator(nbGen, maxdepth))
        print("{:.3f} \t {:.4f}".format(accuracy.mean(), accuracy.std()))
