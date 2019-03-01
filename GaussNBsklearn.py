import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import seaborn as sns; sns.set()

# Importing dataset
data = pd.read_csv("/Users/poorvapatil/Downloads/Employee_data.csv")

# Convert categorical variable to numeric
data["Gender"]=np.where(data["Gender"]=="Male",0,1)
#data["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0,np.where(data["Embarked"]=="C",1,np.where(data["Embarked"]=="Q",2,3)))
# Cleaning dataset of NaN
data=data[["Gender","Age","EstimatedSalary","Purchased"]].dropna(axis=0, how='any')

# Split dataset in training and test datasets
X_train, X_test = train_test_split(data, test_size=0.7, random_state=int(time.time()))


# Instantiate the classifier
gnb = GaussianNB()
used_features = ["Gender","Age","EstimatedSalary"]

# Train classifier
gnb.fit(X_train[used_features].values,X_train["Purchased"])
y_pred = gnb.predict(X_test[used_features])

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Purchased"] != y_pred).sum(),
          100*(1-(X_test["Purchased"] != y_pred).sum()/X_test.shape[0])))



'''

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(X_train.values[:, 0], X_test.values[:, 1], c=y_pred, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim);
    '''
