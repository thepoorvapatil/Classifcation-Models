#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  


filename='/Users/poorvapatil/Downloads/Employee_data.csv'
dataset=pd.read_csv(filename)
dataset["Gender"]=np.where(dataset["Gender"]=="Male",0,1)

X=dataset.iloc[:, 1:-1]
y=dataset.iloc[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)  

svclassifier = SVC(kernel='linear', C=1)  
svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)  


print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  


'''
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = (X_set.values[:, 0]).min() - 1, stop = (X_set.values[:, 0]).max() + 1, step = 0.01),
                     np.arange(start = (X_set.values[:, 1]).min() - 1, stop = (X_set.values[:, 1]).max() + 1, step = 0.01))
plt.contourf(X1, X2, svclassifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = (X_set.values[:, 0]).min() - 1, stop = (X_set.values[:, 0]).max() + 1, step = 0.01),
                     np.arange(start = (X_set.values[:, 1]).min() - 1, stop = (X_set.values[:, 1]).max() + 1, step = 0.01))
plt.contourf(X1, X2, svclassifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
'''
