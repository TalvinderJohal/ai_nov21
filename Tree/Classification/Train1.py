import data_handler1 as dh
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_model():

    x_train, x_test, y_train, y_test, ct, scaler = dh.get_data("/Users/talvinderjohal/Desktop/Talvinder Strive Course/ai_nov21/Tree/Classification/insurance copy.csv")
    dt_clf = DecisionTreeClassifier()
    rf_clf = RandomForestClassifier()

    dt_clf.fit(x_train,y_train)
    rf_clf.fit(x_train,y_train)

    return dt_clf, rf_clf,  ct, scaler

print(train_model())