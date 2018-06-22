# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 21:44:26 2018

@author: saketh
"""
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from mymodel import LinearRegression
model = LinearRegression()

model.loadData(X_train,y_train)

model.fit() #you can also mention no of iterations and learning rate by using
            # model.fit(2500, 0.001)
            #by default iterations = 5000 learening rate = 0.03

y_pred = model.predict(X_test)
print(y_pred)
