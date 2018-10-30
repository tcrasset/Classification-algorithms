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



def compute_accuracy(nb_gen, nb_neighbors, nb_points):
    """Computes the test set accurencies over n generations of the dataset
    for the KNeighborsClassifier class from sklearn.neighbors with a
    particular number of nearest neighbors.

        Parameters
        ----------
        -   nb_gen : number of generations of the dataset.
        -   nb_neighbors : number of nearest neighbors for the KNN model.
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

        estimator = KNeighborsClassifier(n_neighbors = nb_neighbors).fit(X_ls, y_ls)
        y_pred = estimator.predict(X_ts)
        accuracy.append(accuracy_score(y_ts, y_pred))

        if generation == 1:
            plot_boundary("KNN neighbors {}".format(nb_neighbors), estimator, X_ts, y_ts, 0.1)
        
    return accuracy

def compute_cross_val(cv_val, neighbors, nb_points):
    """Computes the optimal value of n_neighbors using a ten-fold cross
    validation strategy.

        Parameters
        ----------
        -   cv_val : number of subsamples in the cross validation.
        -   neighbors : list containing different values for the number
        of nearest neighbors for the KNN model.
        -   nb_points : number of samples.

        Returns
        -------
        -   optimal_nb : the optimal number of nearest neighbors to consider.
        -   MSE : a list of the misclassification errors.
        generations.
    """
    cv_results = []

    X, y = make_dataset2(nb_points, cv_val)
    # Perform 10-fold cross validation
    for nb_neighbors in neighbors:
        knn = KNeighborsClassifier(n_neighbors=nb_neighbors)
        scores = cross_val_score(knn, X, y, cv=cv_val, scoring='accuracy')
        cv_results.append(scores.mean()) # Taking the mean of the cv_val tries
    
    # Compute the misclassification error
    MSE = [1 - x for x in cv_results]

    # Determining the best nb of nearest neighbors
    optimal_nb = neighbors[MSE.index(min(MSE))]

    return (optimal_nb, MSE)

if __name__ == "__main__":
    nb_gen = 5
    nb_points = 1500
    neighbors = [1,5,25,125,625,1200]
    
    for nb_neighbors in neighbors:
        print("N_neighbors : {}".format(nb_neighbors))
        print("Mean \t STD")
        accuracy = np.array(compute_accuracy(nb_gen, nb_neighbors, nb_points))
        print("{:.3f} \t {:.4f}".format(accuracy.mean(), accuracy.std()))

    # Cross-validation testing
    optimal_nb, MSE = compute_cross_val(10, neighbors, nb_points)
    print("The optimal number of neighbors is {}".format(optimal_nb))
    # Plot misclassification error vs k
    plt.plot(neighbors, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()
