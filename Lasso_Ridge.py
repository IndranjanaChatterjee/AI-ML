import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('HousingData.csv')
df.head()
x=df[['RM']]
y=df['MEDV']
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2)
reg=LinearRegression()
reg.fit(train_x,train_y)
reg.score(train_x,train_y)
lasso = linear_model.Lasso()

# Use GridSearchCV to find the best alpha
parameters = {'alpha': [0.1, 1, 10, 100, 1000, 10000]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='r2', cv=5)
lasso_regressor.fit(train_x, train_y)

# Best alpha value
best_alpha = lasso_regressor.best_params_['alpha']
print(f"Best alpha value for Lasso: {best_alpha}")

# Lasso Regression with the best alpha value
lasso_best = linear_model.Lasso(alpha=best_alpha)
lasso_best.fit(train_x, train_y)
lasso_best.score(test_x,test_y)


r=linear_model.Ridge()

rid=linear_model.Ridge(alpha=10)
rid.fit(train_x,train_y)
rid.score(test_x,test_y)






