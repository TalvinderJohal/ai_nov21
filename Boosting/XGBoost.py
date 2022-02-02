import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

data = pd.read_csv("/Users/talvinderjohal/Desktop/Talvinder Strive Course/ai_nov21/Boosting/insurance.csv")

from sklearn.preprocessing import OrdinalEncoder
oc = OrdinalEncoder()
data[["sex", "smoker", "region"]] = oc.fit_transform(data[["sex", "smoker", "region"]])

X = data.iloc[:,0:6].values
y = data.iloc[:,6].values

#data_dmatrix = xgb.DMatrix(data=X, label=y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=0)

xg_reg = xgb.XGBRegressor()

xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)

print(xg_reg.score(X_train, y_train))
print(xg_reg.score(X_test, y_test))