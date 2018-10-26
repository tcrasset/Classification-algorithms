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


if __name__ == "__main__":
    for generation in range(1):
        nbPoints = 1500
        X, y = make_dataset2(nbPoints)


        numberIterations = 15
        accuracy_arr = np.empty((numberIterations,5))
        
        for i in range(0,numberIterations):
            X_ls, X_ts, y_ls, y_ts = train_test_split(X, y , train_size = 1200/1500)
            
            estimator_maxdepth_1 = DecisionTreeClassifier(max_depth=1).fit(X_ls,y_ls)
            estimator_maxdepth_2 = DecisionTreeClassifier(max_depth=2).fit(X_ls,y_ls)
            estimator_maxdepth_4 = DecisionTreeClassifier(max_depth=4).fit(X_ls,y_ls)
            estimator_maxdepth_8 = DecisionTreeClassifier(max_depth=8).fit(X_ls,y_ls)
            estimator_maxdepth_None = DecisionTreeClassifier().fit(X_ls,y_ls)    

            estimators = [
                estimator_maxdepth_1,
                estimator_maxdepth_2,
                estimator_maxdepth_4,
                estimator_maxdepth_8,
                estimator_maxdepth_None
            ]
            for estimator, j in zip(estimators,range(5)):
                y_pred = estimator.predict(X_ts)
                accuracy_arr[i][j] = accuracy_score(y_ts, y_pred)

        accuracy_arr = accuracy_arr.T #transpose array

        labels = [1,2,4,8,"None"]
        print("Generation {}".format(generation))
        print("Number of iterations: {}".format(numberIterations))
        print("Estimator \t Mean \t STD")

        for label, i in zip(labels,range(5)):
            print("{:>10} \t {:.3f} \t {:.4f}".format(label,accuracy_arr[i].mean(),accuracy_arr[i].std()))
            plot_boundary("DT maxdepth {}".format(label),estimators[i], X_ts, y_ts,0.1)
