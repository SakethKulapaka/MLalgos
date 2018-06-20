# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 22:33:01 2018

@author: saketh
linear regression model from scratch
i.e, using numpy and pandas(maybev t)

to do :
    :>make a linear regresor class
    :>functions :
        :> split the dataset - use scikit-learn
        :> costFunction
        :> hypothesis y = b1x1 + b0
        :> gradient descent
    :> assign random value to theta/weights
"""

import numpy as np

class myLinearRegressionModel :
    
    def __init__(self) :
        self.iter = 5000
        self.alpha = 0.1
        self.hyp = 0
        self.loss = 0
        self.theta = [0.69, 1.05]
        
    def loadData(self,X_train,y_train):
        self.y = y_train

        u = np.mean(X_train)
        r = np.ptp(X_train, axis = 0)
        self.X1= (X_train-u)/r
        
        self.m = len(X_train)
        x0 = np.array([1]*self.m)
        x0 = x0.reshape(self.m,1)
        self.X = np.concatenate((x0, X_train), 
                                      axis=1)        
    def hypothesis(self) :
        #self.hyp = self.theta[0] * X_train + self.theta[1] 
        self.hyp = np.matmul( self.X, np.array(self.theta).T)
        #print(self.hyp)
        
    def costfunc(self) :
        temp = 0
        temp = np.sum((self.hyp-self.y)**2)
        temp *= (1/(2*self.m))
        self.loss = temp
    
    def gradients(self) :
            temp0 =  (np.sum(self.hyp-self.y))*(self.alpha/self.m)
            temp1 =  (np.sum((self.hyp-self.y)*self.X1))*(self.alpha/self.m)
            temp0 = self.theta[0] - temp0
            temp1 = self.theta[1] - temp1
            self.theta[0] = temp0
            self.theta[1] = temp1
            print("t0 : ",self.theta[0]," t1 : ",self.theta[1])
            
    def fit(self) :
        for i in range(self.iter) :
            print('iteration no: ',i+1)
            self.hypothesis()
            self.costfunc()
            print('loss : ',self.loss)
            self.gradients()
    
    def predict(self,X_test) :
        u = np.mean(X_test)
        r = np.ptp(X_test, axis = 0)
        test= (X_test-u)/r
        n = len(X_test)
        x0 = np.array([1]*n)
        x0 = x0.reshape(n,1)
        test = np.concatenate((x0, test), 
                                      axis=1) 
        
        pred = np.matmul( test, np.array(self.theta).T)
        return pred
        
        
    
        
        
        
    
    
    