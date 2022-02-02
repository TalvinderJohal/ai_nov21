import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

def get_data(path):
    
    data = pd.read_csv(path)
    
    x, y = data.values[:,:-1], data.values[:,-1]
    
    x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.3, random_state= 42)
    
    c_transformer = ColumnTransformer([("ordinal",OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value=-1),[1,4,5]),('non_transformed','passthrough',[0,2,3])])
    
    x_train = c_transformer.fit_transform(x_train)
    x_test = c_transformer.transform(x_test)
    
    # Scaling is not mandatory for TREE ALGOs. BUT BEST PRACTICE
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    return x_train_scaled, x_test_scaled, y_train, y_test, c_transformer, scaler 
