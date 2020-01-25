#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 02:52:11 2020

@author: darp_lord
"""
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score


def evaluate(clf, test_X, test_y):
    preds=clf.predict(test_X)
    print(classification_report(test_y.values, preds))
    print("Accuracy:",accuracy_score(test_y.values, preds))

def TrainClassifiers(train_X, test_X, train_y, test_y):
	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
	         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
	         "Naive Bayes", "QDA"]
	
	classifiers = [
	    KNeighborsClassifier(3),
	    SVC(kernel="linear", C=0.025),
	    SVC(gamma=2, C=1),
	    GaussianProcessClassifier(1.0 * RBF(1.0)),
	    DecisionTreeClassifier(),
	    RandomForestClassifier(),
	    MLPClassifier(),
	    AdaBoostClassifier(),
	    GaussianNB(),
	    QuadraticDiscriminantAnalysis()]
	
	for name, clf in zip(names, classifiers):
	    clf.fit(train_X, train_y)
	    print(name)
	    evaluate(clf, test_X, test_y)
	    print()