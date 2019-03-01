#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix  



filename='/Users/poorvapatil/Downloads/Employee_data.csv'
dataset=pd.read_csv(filename)
dataset["Gender"]=np.where(dataset["Gender"]=="Male",0,1)


#preprocessing into attributes-X and label-Y
X=dataset.iloc[:, 1:-1]
Y=dataset.iloc[:, 4]

#train and test data split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)  

#scale the training data
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  


#make predictions on training data
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, Y_train)  

#make predictions on the test data
Y_pred = classifier.predict(X_test)  

#print
print(confusion_matrix(Y_test, Y_pred))  
print(classification_report(Y_test, Y_pred))  


error = []

# Calculating error for K values between 1 and 50
for i in range(1, 50):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != Y_test))
    
#plot error values against K values
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  



