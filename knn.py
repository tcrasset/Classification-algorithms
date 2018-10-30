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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from plot import plot_boundary

# (Question 2)

# Put your funtions here
# ...

def trainEstimator(nbGen, neighbors):
    accuracy_arr = []
    for generation in range(nbGen):
        nbPoints = 1500
        seed = generation
        X, y = make_dataset2(nbPoints, seed)
        X_ls, X_ts, y_ls, y_ts = train_test_split(X, y, train_size = 1200, test_size = 300)

        estimator = KNeighborsClassifier(n_neighbors = neighbors).fit(X_ls,y_ls)

        y_pred = estimator.predict(X_ts)
        accuracy_arr.append(accuracy_score(y_ts, y_pred))

        if generation == 1:
            plot_boundary("KNN neighbors {}".format(neighbors),estimator , X_ts, y_ts, 0.1)
        
    return accuracy_arr


def crossval(k_list, cv_val):
    nbPoints = 1500
    seed = cv_val
    cv_results = []

    X, y = make_dataset2(nbPoints, seed)

    # perform 10-fold cross validation
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=cv_val, scoring='accuracy')
        cv_results.append(scores.mean()) #Taking the mean of the cv_val tries

    return cv_results

if __name__ == "__main__":
    nbGen = 5
    neighbors = [1,5,25,125,625,1200]
    
    
    for neighbor in neighbors:
        print("N_neighbors : {}".format(neighbor))
        print("Mean \t STD")
        accuracy = np.array(trainEstimator(nbGen, neighbor))
        print("{:.3f} \t {:.4f}".format(accuracy.mean(), accuracy.std()))

    cv_results = crossval(neighbors, 10) # Cross-validation testing

    # changing to misclassification error
    MSE = [1 - x for x in cv_results]

    # determining best k
    optimal_k = neighbors[MSE.index(min(MSE))]
    print("The optimal number of neighbors is {}".format(optimal_k))

    # plot misclassification error vs k
    plt.plot(neighbors, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()

