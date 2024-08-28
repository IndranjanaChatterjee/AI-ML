import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('headbrain.csv')
x=dataset['Head Size(cm^3)'].values
y=dataset['Brain Weight(grams)'].values
mean_x=np.mean(x)
mean_y=np.mean(y)
numer=0
denom=0
for i in range(len(x)):
    numer+=(x[i]-mean_x)*(y[i]-mean_y)
    denom+=(x[i]-mean_x)**2
m=numer/denom
c=mean_y-(m*mean_x)
max_x=np.max(x)+100
min_x=np.min(x)-100
x1=np.linspace(min_x,max_x,100)
y1=(m*x1)+c
plt.plot(x1,y1,color="blue",label="Regression Line")
plt.scatter(x,y,color="magenta",label="data points")
x_test=int(input("Give the head size:"))
y_predicted=(m*x_test)+c
print(x)
