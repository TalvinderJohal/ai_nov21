import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, url):
        self.url = url
    
    def call_data(self):
        dataset = pd.read_csv(self.path)
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
    
    def sigmoid(slef, z):
        return 1/(1 + np.exp(-z))

    def __getCoefficients(self,X,y):
        xDotx = np.dot(X.T,X)
        xDotxInverse = np.linalg.inv(xDotx)
        xDotxInverseDotXT = np.dot(xDotxInverse,X.T)
        return np.dot(xDotxInverseDotXT,y)
