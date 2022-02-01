import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/talvinderjohal/Desktop/Talvinder Strive Course/insurance.csv")

from sklearn.preprocessing import OrdinalEncoder
oc = OrdinalEncoder()
data[["sex", "smoker", "region"]] = oc.fit_transform(data[["sex", "smoker", "region"]])

X = data.iloc[:,0:6].values
y = data.iloc[:,6].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=0)

from sklearn.ensemble import AdaBoostRegressor
abc = AdaBoostRegressor(random_state=0)
abc.fit(X_train, y_train)

print(abc.score(X_train, y_train))

print(abc.score(X_test, y_test))

