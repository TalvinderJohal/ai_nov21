import Data_Handler as dh
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_model():

    x_train, x_test, y_train, y_test, scaler = dh.get_data("/Users/talvinderjohal/Desktop/Talvinder Strive Course/Hippocratia Challenge/heart.csv.xls")

    clf_1 = SVC().fit(x_train, y_train)
    clf_2 = LogisticRegression().fit(x_train, y_train)
    clf_3 = RandomForestClassifier(n_estimators=100, max_depth=4).fit(x_train, y_train)
    clf_4 = KNeighborsClassifier(n_neighbors=4).fit(x_train, y_train)
    clf_5 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=1).fit(x_train, y_train)

    return clf_1, clf_2, clf_3, clf_4, clf_5, scaler

# print(train_model())