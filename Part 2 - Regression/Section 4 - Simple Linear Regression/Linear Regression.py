#Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#splitting the dataset in training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=1/3)

# fitting simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#predict
y_pred = regressor.predict(X_test)

# visualising the training set results
plt.scatter(X_train,Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()