# Step 1: Preprocessing
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing data
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelEncoder_X1 = LabelEncoder()
X[:,1] = labelEncoder_X1.fit_transform(X[:,1])
labelEncoder_X2 = LabelEncoder()
X[:,2] = labelEncoder_X2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# split into training n test
from sklearn.model_selection import train_test_split
X_train,  X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
