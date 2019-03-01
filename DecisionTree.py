#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:13:04 2019

@author: poorvapatil
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt


filename='/Users/poorvapatil/Downloads/Employee_data.csv'
dataset=pd.read_csv(filename)
dataset["Gender"]=np.where(dataset["Gender"]=="Male",0,1)

X=dataset.iloc[:, 1:-1] #predictor variables
Y=dataset.iloc[: , 4] #outcome variable

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)



clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

print("prediction using gini index criteria")
y_pred = clf_gini.predict(X_test)
print(y_pred)
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)

print()

print("prediction using information gain criteria")
y_pred_en = clf_entropy.predict(X_test)
print(y_pred_en)
print ("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)


