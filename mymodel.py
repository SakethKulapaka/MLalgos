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
        
class LinearRegression :
    
    def __init__(self) :
        self.alpha = 0.03
        self.iter = 5000
        self.hyp= 0
        self.loss = 0
        
    def loadData(self, X_train, y_train) :
        self.m, self.n = np.shape(X_train)
        self.theta = np.array([[np.random.randn()]*(self.n+1)]).reshape(1,self.n+1)
        
        x0 = np.array([1]*self.m).reshape(self.m,1)
        self.X = np.concatenate((x0,X_train), axis=1)
        self.X = self.X/np.amax(self.X, axis=0)
        #print(self.X)
        
        self.y = y_train.reshape(self.m,1)
        
    def calcHyp(self) :
        self.hyp = np.matmul(self.X,self.theta.T).reshape(self.m,1)
        #print(self.hyp)
        
    def costfunc(self) :
        self.loss = ((self.hyp-self.y)**2)/(2*self.m)
        self.loss = np.sum(self.loss)
        
    def gradients(self) :
        temp = (self.hyp-self.y).reshape(self.m,1)
        temp = np.sum(self.X*temp, axis=0)
        self.theta -= ((temp*self.alpha)/self.m)
        #print(self.theta)
        
    def fit(self,iter = 5000,alpha=0.03) :
        self.iter = iter
        self.alpha = alpha
        for i in range(self.iter) :
            print('iteration no: ',i+1)
            self.calcHyp()
            self.costfunc()
            print('loss : ',self.loss)
            self.gradients()
    
    def predict(self,X_test) :
        test= X_test/np.max(X_test, axis=0)
        z = len(X_test)
        x0 = np.array([1]*z)
        x0 = x0.reshape(z,1)
        test = np.concatenate((x0, test), 
                                      axis=1) 
        
        pred = np.matmul( test, self.theta.T)
        return pred
        
        
        
    
    
    