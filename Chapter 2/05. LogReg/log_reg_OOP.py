import numpy as np
import pandas as pd

class LogisticRegression:

    def __init__(self,iterations,alpha):
        self.iterations = iterations
        self.alpha = self.alpha
    
    def sigmoid(self, z):
        return(1/(1+np.exp(-z)))
    
    def fit(self,x,y):
        m=x.shape[0]
        w=np.random.randn(shape[1],1)

        cost_= []
        for i in len(iterations):
            a = np.dot(x,w)
            z=self.sigmoid(a)

            cost = (-1/m) *(np.dot(y,np.log(z))+(np.dot((1-y),np.log(1-z))))
            cost_.append[cost]
            dw = (1/m)*np.dot(x.T,(z-y))

            w=w-(self.alpha*dw)

        return self
    def predict(self,x,threshold):
        probability=self.sigmoid(np.dot(x,self.w))
        if(probability>threshold):
            return(1,probability)
        else:
            return(0,probability)
