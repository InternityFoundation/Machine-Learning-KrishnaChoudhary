# Step 1: Preprocessing
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing data
dataset = pd.read_csv("Wine.csv")
X = dataset.iloc[:,0:13].values
Y = dataset.iloc[:,13].values

# split into training n test
from sklearn.model_selection import train_test_split
X_train,  X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train,Y_train)
X_test = lda.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators=10,criterion='entropy')
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)

from sklearn.metrics import confusion_matrix
cn = confusion_matrix(Y_test,Y_pred)    