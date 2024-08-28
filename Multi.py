import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math

df=pd.read_csv('homeprices.csv')
data=math.floor(df.bedrooms.median())
df.bedrooms=df.bedrooms.fillna(data)
red=LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
reg.predict([[2500,4,5]])
