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
        self.iter = 2000
        self.alpha = 0.01
        self.hyp = 0
        self.loss = 0
        self.theta = [np.random.randn(),np.random.randn()]
        
    def loadData(self,X_train,y_train):
        self.y = y_train

        #u = np.mean(X_train)
        #r = np.ptp(X_train, axis = 0)
        self.X1= X_train/np.max(X_train)
        
        self.m = len(X_train)
        x0 = np.array([1]*self.m)
        x0 = x0.reshape(self.m,1)
        self.X = np.concatenate((x0, X_train), 
                                      axis=1)     
        print(self.X)
        print(self.y)
    def hypothesis(self) :
        #self.hyp = self.theta[0] * X_train + self.theta[1] 
        self.hyp = np.matmul( self.X, np.array(self.theta).reshape(2,1))
        #print(self.hyp)
        
    def costfunc(self) :
        temp = 0
        temp = np.sum((self.hyp-self.y)**2)
        temp *= (1/(2*self.m))
        self.loss = temp
        print(self.loss)

    def gradients(self) :
            #temp0 =  (np.sum(self.hyp-self.y))*(self.alpha/self.m)
            #temp1 =  (np.sum(self.X1*(self.hyp-self.y)))*(self.alpha/self.m)
            #temp0 = self.theta[0] - temp0
            #temp1 = self.theta[1] - temp1
            temp = np.sum(self.X*(self.hyp-self.y),axis=0)*(self.alpha/self.m)
            self.theta[0] = temp[0]
            self.theta[1] = temp[1]
            print("t0 : ",self.theta[0]," t1 : ",self.theta[1])
            
    def fit(self) :
        for i in range(self.iter) :
            print('iteration no: ',i+1)
            self.hypothesis()
            self.costfunc()
            print('loss : ',self.loss)
            self.gradients()
    
    def predict(self,X_test) :
        #u = np.mean(X_test)
        #r = np.ptp(X_test, axis = 0)
        test= X_test/np.max(X_test)
        n = len(X_test)
        x0 = np.array([1]*n)
        x0 = x0.reshape(n,1)
        test = np.concatenate((x0, test), 
                                      axis=1) 
        
        pred = np.matmul( test, np.array(self.theta).reshape(2,1))
        return pred
        
class LinearRegression :
    
    def __init__(self) :
        self.alpha = 0.03
        self.iter = 2000
        self.hyp= 0
        self.loss = 0
        
    def loadData(self, X_train, y_train) :
        self.m, self.n = np.shape(X_train)
        self.theta = np.array([[np.random.randn()]*(self.n+1)]).reshape(1,self.n+1)
        
        x0 = np.array([1]*self.m).reshape(self.m,1)
        self.X = np.concatenate((x0,X_train), axis=1)
        self.X = self.X/np.amax(self.X, axis=0)
        print(self.X)
        
        self.y = y_train.reshape(self.m,1)
        print(np.shape(self.y))
    def calcHyp(self) :
        self.hyp = np.matmul(self.X,self.theta.T).reshape(self.m,1)
        print(self.hyp)
        
    def costfunc(self) :
        self.loss = ((self.hyp-self.y)**2)/(2*self.m)
        self.loss = np.sum(self.loss)
        
    def gradients(self) :
        temp = (self.hyp-self.y).reshape(self.m,1)
        temp = np.sum(self.X*temp, axis=0)
        self.theta -= ((temp*self.alpha)/self.m)
        print(self.theta)
        #temp = np.sum(self.X*temp,axis =0)*(self.alpha/self.m)
        #temp = temp.reshape(1,self.n)
        #self.theta = self.theta - temp
    
    def fit(self) :
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
        
        
        
    
    
    