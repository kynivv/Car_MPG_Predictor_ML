# Libraries
import pandas as pd
import numpy as np

from sklearn.metrics import explained_variance_score as evs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


# Data Import
data_path = 'Automobile.csv'

data = pd.read_csv(data_path)


# EDA & Preprocessing
print(data.shape)
print(data.info())

print(data.isnull().sum())

data = data.dropna()

print(data.info())

print(data.isnull().sum())

for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

print(data.dtypes)

data = data.drop('name', axis= 1)

print(data)


# Training Test Data Split
X = data.drop('mpg', axis= 1)
Y = data['mpg']

print(X.info(), Y.info())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size= 0.2,
                                                    random_state= 24
                                                    )


# Creating Model

models = [RandomForestRegressor(),
          AdaBoostRegressor(),
          DecisionTreeRegressor()
          ]

for m in models:
    print(m)

    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuracy : {evs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy : {evs(Y_test, pred_test)}/n')