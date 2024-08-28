import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
boston=pd.read_csv('HousingData.csv')
z=pd.DataFrame(boston.corr().round(2))
x=boston['RM']
y=boston['MEDV']

x=pd.DataFrame(x)
y=pd.DataFrame(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape)
linearRegressionClassifier=LinearRegression()
linearRegressionClassifier.fit(x_train,y_train)
y_pred=linearRegressionClassifier.predict(x_test)
mean_sq=np.sqrt(mean_squared_error(y_test,y_pred))
r2_square=linearRegressionClassifier.score(x_test,y_test)
print(r2_square)
