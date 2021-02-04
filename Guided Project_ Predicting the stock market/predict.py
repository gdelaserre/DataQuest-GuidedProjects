from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# import file, transform date to datetime and sort values from oldest to most recent
sp = pd.read_csv('sphist.csv')
sp['Date'] = pd.to_datetime(sp['Date'])
sp.sort_values(['Date'], ascending = True, inplace = True)

# Create indicators
sp['avg_5'] = sp['Close'].rolling(5).mean().shift(1)
sp['avg_30'] = sp['Close'].rolling(30).mean().shift(1)
sp['avg_365'] = sp['Close'].rolling(365).mean().shift(1)
sp['ratio_avg_5_365'] = sp['avg_5'] / sp['avg_365']

sp['std_5'] = sp['Close'].rolling(5).std().shift(1)
sp['std_30'] = sp['Close'].rolling(30).std().shift(1)
sp['std_365'] = sp['Close'].rolling(365).std().shift(1)
sp['ratio_std_5_365'] = sp['std_5'] / sp['std_365']

# remove rows before 1951-01-03 & drop NaN values
sp = sp[sp['Date'] >= '1951-01-03']
sp.dropna(axis = 0, inplace = True)

#split between train and test
train = sp[sp['Date'] < '2013-01-01']
test = sp[sp['Date'] >= '2013-01-01']

# Train a linear regression model, using the train Dataframe
lr = LinearRegression()
feature_cols = sp.columns.drop(['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close', 'Date'])

X_train = train[feature_cols]
y_train = train.Close

X_test = test[feature_cols]
y_test = test.Close

lr.fit(X_train, y_train)
predictions = lr.predict(X_test)

mae = mean_absolute_error(predictions, y_test)
mse = mean_squared_error(predictions, y_test)

# Check first model
print('MAE = {}'.format(mae))
print('MSE = {}'.format(mse))

# Rework model to predict only one day ahead

train = sp.iloc[:-1]
test = sp.iloc[-1:,:]
lr = LinearRegression()
feature_cols = sp.columns.drop(['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close', 'Date'])

X_train = train[feature_cols]
y_train = train.Close

X_test = test[feature_cols]
y_test = test.Close

lr.fit(X_train, y_train)
predictions = lr.predict(X_test)

mae = mean_absolute_error(predictions, y_test)
mse = mean_squared_error(predictions, y_test)

print('New MAE = {}'.format(mae))
print('New MSE = {}'.format(mse))



