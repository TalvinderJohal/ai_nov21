# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

class LogisticRegressionModel:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def prediction(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.25, random_state = 0)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
    
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        return(cm)
