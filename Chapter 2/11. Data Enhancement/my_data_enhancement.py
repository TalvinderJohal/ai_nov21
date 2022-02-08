import numpy    as np
from numpy.testing._private.utils import decorate_methods
import pandas   as pd
import seaborn  as sb
import matplotlib.pyplot as plt
import sklearn  as skl
import time

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config

from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor
from sklearn.ensemble      import ExtraTreesRegressor
from sklearn.ensemble      import AdaBoostRegressor
from sklearn.ensemble      import GradientBoostingRegressor
from xgboost               import XGBRegressor
from lightgbm              import LGBMRegressor
from catboost              import CatBoostRegressor


np.random.seed(0)

df = pd.read_csv("/Users/talvinderjohal/Desktop/Talvinder Strive Course/ai_nov21/Chapter 2/11. Data Enhancement/data/london_merged.csv")
df_copy = df.copy()

df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

df_copy["year"] = df_copy['timestamp'].dt.year
df_copy["month"] = df_copy['timestamp'].dt.month
df_copy["day_of_month"] = df_copy['timestamp'].dt.day
df_copy["day_of_week"] = df_copy['timestamp'].dt.weekday
df_copy["hour"] = df_copy['timestamp'].dt.hour
df_copy.drop('timestamp', axis=1, inplace=True)

# print(df_copy.head())
# print(df_copy.info())

def data_enhancement(df_copy):
    new_data = df_copy

    for code in df_copy['weather_code'].unique():
        coded_weather =  new_data[new_data['season'] == code]
        hum_mean = coded_weather['hum'].mean()
        wind_speed_mean = coded_weather['wind_speed'].mean()
        t1_mean = coded_weather['t1'].mean()
        t2_mean = coded_weather['t2'].mean()

        for i in new_data[new_data["weather_code"] == code].index:
            if np.random.randint(2) == 1:
                new_data['hum'].values[i] += hum_mean/20
            else:
                new_data['hum'].values[i] -= hum_mean/20
                
            if np.random.randint(2) == 1:
                new_data['wind_speed'].values[i] += wind_speed_mean/20
            else:
                new_data['wind_speed'].values[i] -= wind_speed_mean/20
                
            if np.random.randint(2) == 1:
                new_data['t1'].values[i] += t1_mean/20
            else:
                new_data['t1'].values[i] -= t1_mean/20
                
            if np.random.randint(2) == 1:
                new_data['t2'].values[i] += t2_mean/20
            else:
                new_data['t2'].values[i] -= t2_mean/20
    return new_data

# print(df_copy.head(3))
gen = data_enhancement(df_copy)
# print(gen.head(3) )

y = df_copy['cnt']
x = df_copy.drop(['cnt'], axis=1)

cat_vars = ['season','is_weekend','is_holiday','year','month','weather_code']
num_vars = ['t1','t2','hum','wind_speed']


x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y,
                                    test_size=0.2,
                                    random_state=0  # Recommended for reproducibility
                                )



extra_sample = gen.sample(gen.shape[0] // 3)
x_train = pd.concat([x_train, extra_sample.drop(['cnt'], axis=1 ) ])
y_train = pd.concat([y_train, extra_sample['cnt'] ])
# print(x_train)
# print(y_train)


transformer = preprocessing.PowerTransformer()
y_train = transformer.fit_transform(y_train.values.reshape(-1,1))
y_val = transformer.transform(y_val.values.reshape(-1,1))

# print(x_train)
# print(y_val)


rang = abs(y_train.max()) + abs(y_train.min())
# print(rang)

num_4_treeModels = pipeline.Pipeline(steps=[
    ('imputer', impute.SimpleImputer(strategy='constant', fill_value=-9999)),
])

# cat_4_treeModels = pipeline.Pipeline(steps=[
#     ('imputer', impute.SimpleImputer(strategy='constant', fill_value='missing')),
#     ('ordinal', preprocessing.OrdinalEncoder()) # handle_unknown='ignore' ONLY IN VERSION 0.24
# ])

tree_prepro = compose.ColumnTransformer(transformers=[
    ('num', num_4_treeModels, num_vars),
#    ('cat', cat_4_treeModels, cat_vars),
], remainder='drop') # Drop other vars not specified in num_vars or cat_vars

tree_classifiers = {
  "Decision Tree": DecisionTreeRegressor(),
  "Extra Trees":   ExtraTreesRegressor(n_estimators=100),
  "Random Forest": RandomForestRegressor(n_estimators=100),
  "AdaBoost":      AdaBoostRegressor(n_estimators=100),
  "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
  "XGBoost":       XGBRegressor(n_estimators=100),
  "LightGBM":      LGBMRegressor(n_estimators=100),
  "CatBoost":      CatBoostRegressor(n_estimators=100),
}
### END SOLUTIONv

tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}

results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], " % error": [], 'Time': []})

for model_name, model in tree_classifiers.items():
    
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time
        
    pred = model.predict(x_val)
    
    results = results.append({"Model":    model_name,
                              "MSE": metrics.mean_squared_error(y_val, pred),
                              "MAB": metrics.mean_absolute_error(y_val, pred),
                              " % error": metrics.mean_squared_error(y_val, pred) / rang,
                              "Time":     total_time},
                              ignore_index=True)
### END SOLUTION


results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
results_ord.index += 1 
results_ord.style.bar(subset=['MSE', 'MAE'], vmin=0, vmax=100, color='#5fba7d')

print(results_ord)


print(y_train.max())
print(y_train.min())
print(y_val[3])
print(tree_classifiers['Random Forest'].predict(x_val)[3])
