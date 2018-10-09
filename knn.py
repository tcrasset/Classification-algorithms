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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# (Question 2)

# Put your funtions here
# ...


if __name__ == "__main__":
    nbPoints = 1500
    seed = 666
    X, y = make_dataset2(nbPoints, seed)
    
    X_ls, X_ts, y_ls, y_ts = train_test_split(X, y , train_size = 1200/1500)

    estimator_n_neighbors_1 = KNeighborsClassifier(n_neighbors = 1).fit(X_ls,y_ls)
    estimator_n_neighbors_5 = KNeighborsClassifier(n_neighbors = 5).fit(X_ls,y_ls)
    estimator_n_neighbors_25 = KNeighborsClassifier(n_neighbors = 25).fit(X_ls,y_ls)
    estimator_n_neighbors_125 = KNeighborsClassifier(n_neighbors= 125).fit(X_ls,y_ls)
    estimator_n_neighbors_625 = KNeighborsClassifier(n_neighbors= 625).fit(X_ls,y_ls)
    estimator_n_neighbors_1200 = KNeighborsClassifier(n_neighbors= 1200).fit(X_ls,y_ls)


    estimators = [
        estimator_n_neighbors_1,
        estimator_n_neighbors_5,
        estimator_n_neighbors_25,
        estimator_n_neighbors_125,
        estimator_n_neighbors_625,
        estimator_n_neighbors_1200
    ]

    accuracy_arr = np.empty((6))

    for estimator, j in zip(estimators,range(6)):
        y_pred = estimator.predict(X_ts)
        accuracy_arr[j] = accuracy_score(y_ts, y_pred)
    
    labels = [1,5,25,125,625,1200]
    print("N_neighbors \t Accuracy score")

    for label, i in zip(labels,range(6)):
        print("{:>10} \t {:.3f}".format(label,accuracy_arr[i]))

