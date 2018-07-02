#Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing data
dataset = pd.read_csv("train.csv")
X = dataset.iloc[:,2:].values
Y = dataset.iloc[:,1].values

# split into training n test
from sklearn.model_selection import train_test_split
X_train,  X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=3558)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((4459,1)).astype(int), values=X, axis=1)
X_opt = X[:,:]
regressor_ols = sm.OLS(endog=Y, exdog =X_opt).fit()
regressor_ols.summary()