# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 21:44:26 2018

@author: saket
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from mymodel import myLinearRegressionModel
model = myLinearRegressionModel()
model.loadData(X_train,y_train)
model.fit()
y_pred = model.predict(X_test)
