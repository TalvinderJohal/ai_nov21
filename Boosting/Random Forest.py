import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# Preprocessing

data = pd.read_csv("/Users/talvinderjohal/Desktop/Talvinder Strive Course/ai_nov21/Boosting/insurance.csv")

def encoder():

    from sklearn.preprocessing import OrdinalEncoder
    oc = OrdinalEncoder()
    data[["sex", "smoker", "region"]] = oc.fit_transform(data[["sex", "smoker", "region"]])

    X = data.iloc[:,0:6].values
    y = data.iloc[:,6].values
    return X, y

def splitter():

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(encoder()[0], encoder()[1], test_size= 0.25, random_state=0)
    return X_train, X_test, y_train, y_test

def rfg():
    from sklearn.ensemble import RandomForestRegressor
    rfg = RandomForestRegressor(n_estimators=100)
    rfg.fit(splitter()[0], splitter()[2])
    return(rfg.score(splitter()[0], splitter()[2])), (rfg.score(splitter()[1], splitter()[3]))

def abc():

    from sklearn.ensemble import AdaBoostRegressor
    abc = AdaBoostRegressor(random_state=0)
    abc.fit(splitter()[0], splitter()[2])
    return(abc.score(splitter()[0], splitter()[2]), abc.score(splitter()[1], splitter()[3]))

# def xgb():
#     xg_reg = xgb.XGBRegressor()

#     xg_reg.fit(splitter()[0],splitter()[2])
#     preds = xg_reg.predict(splitter()[1])

#     print(xg_reg.score(splitter()[0], splitter()[2]))
#     print(xg_reg.score(splitter()[1], splitter()[3]))





encoder()
splitter()
print(f"Accuracy score for tain and test data for Random Forest Regressor:   {rfg()}")
print(f"Accuracy score for tain and test data for AdaBOost Regressor:   {abc()}")
# print(f"Accuracy score for tain and test data for XGB Regressor:   {xgb()}")

