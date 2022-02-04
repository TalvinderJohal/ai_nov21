import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_data(path):

    data = pd.read_csv(path)

    X = data.iloc[:,:-1].values
    y = pd.DataFrame(data["output"]).values.flatten()
    
    x_train, x_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state= 0)
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler

# print(get_data("/Users/talvinderjohal/Desktop/Talvinder Strive Course/Hippocratia Challenge/heart.csv.xls"))
