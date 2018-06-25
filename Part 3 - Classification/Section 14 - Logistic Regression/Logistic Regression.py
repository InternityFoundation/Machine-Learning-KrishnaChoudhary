#Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,2:4].values
Y = dataset.iloc[:,-1].values

#splitting the dataset in training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

#fitting logistic regression 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

#predict
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cn = confusion_matrix(Y_test,y_pred)