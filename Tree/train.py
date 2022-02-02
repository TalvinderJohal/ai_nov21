import data_handler as dh
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
#from catboost import CatBoostRegressor
#dt
#rf
#choose

def train_model():

    x_train, x_test, y_train, y_test, ct, scaler = dh.get_data("/Users/talvinderjohal/Desktop/Talvinder Strive Course/ai_nov21/Tree/insurance.csv")

    clf_1 = GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000)
    clf_2 = XGBRegressor()
    #clf_3 = CatBoostRegressor(max_depth=4, verbose=False)

    clf_1.fit(x_train,y_train)
    clf_2.fit(x_train,y_train)
    #clf_3.fit(x_train,y_train)


    return clf_1, clf_2,  ct, scaler

#clf_3,