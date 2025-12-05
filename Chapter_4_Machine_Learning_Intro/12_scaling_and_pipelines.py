"""
Converted from: Chapter 4 - 12 - Scaling and pipelines.ipynb

This script contains code extracted from a Jupyter notebook.
Code has been organized for modular execution.
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [7.0, 4.5]
plt.rcParams['figure.dpi'] = 150

data = pd.read_csv('stock_data_performance_fundamentals_300.csv', index_col=0)
x = data.drop(columns='Perf')
y = data['Perf']

"""
First Standard Scaler
"""

x.describe()

myKey = x.keys()[2]
x[myKey].hist(bins=50)

scaler = StandardScaler() # Create Standard Scaler Object
x_s = scaler.fit_transform(x) # Fit the scaler, returns a numpy array
x_s = pd.DataFrame(x_s, columns=x.keys())

x_s.describe()

x_s[myKey].hist(bins=50)

# Side-by-side

myKey = x.keys()[2]# Can change key number, 2 is P/E ratios

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
x[myKey].hist(bins=40)
plt.title('Non Scaled {}, Mean {}, std. dev. {}'.format(myKey, 
                                            round(x[myKey].mean(),2), 
                                            round(x[myKey].std(),2)))
plt.xlim([-400, 400]);

plt.subplot(1,2,2)
x_s[myKey].hist(bins=40)
plt.title('Standard Scaler {}, Mean {}, std. dev. {}'.format(myKey, 
                                            round(x_s[myKey].mean(),2), 
                                            round(x_s[myKey].std(),2)))
plt.xlim([-10, 10]);

# Sometimes need more than just transform

myKey = x.keys()[4]#P/S ratios

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
x[myKey].hist(bins=100)
plt.title('Non Scaled {}, Mean {}, std. dev. {}'.format(myKey, round(x[myKey].mean(),2), round(x[myKey].std(),2)))
#plt.xlim([-400, 400]);

plt.subplot(1,2,2)
x_s[myKey].hist(bins=100)
plt.title('Standard Scaler {}, Mean {}, std. dev. {}'.format(myKey, round(x_s[myKey].mean(),2), round(x_s[myKey].std(),2)))
#plt.xlim([-10, 10]);

"""
Now Power Transformer
"""

transformer = PowerTransformer()
x_t=transformer.fit_transform(x)

x_t = pd.DataFrame(x_t)
x_t.columns = x.keys()
x_t.describe()

x_t[myKey].hist(bins=100)

"""
Have a look at scaling and transforming
"""

myKey = x.keys()[4] #4

plt.figure(figsize=(7,12))
plt.subplot(3,1,1)
x[myKey].hist(bins=40)
plt.title('Non Scaled {}, Mean {}, std. dev. {}'.format(myKey, 
                                                        round(x[myKey].mean(),2), 
                                                        round(x[myKey].std(),2)),
         fontsize=15)

plt.subplot(3,1,2)
x_s[myKey].hist(bins=40)
plt.title('Standard Scaled {}, Mean {}, std. dev. {}'.format(myKey, 
                                                             round(x_s[myKey].mean(),2), 
                                                             round(x_s[myKey].std(),2)),
          
         fontsize=15)

plt.subplot(3,1,3)
x_t[myKey].hist(bins=40)
plt.title('Power Transformed {}, Mean {}, std. dev. {}'.format(myKey, 
                                                               round(x_t[myKey].mean(),2), 
                                                               round(x_t[myKey].std(),2)),
         fontsize=15)

"""
Pipeline Example
"""
# To use a transformer with Scikit-Learn, it is easier to assemble your transformations into a pipeline.

data = pd.read_csv('stock_data_performance_fundamentals_300.csv', index_col=0) # Read in our data
X=data.drop(columns='Perf')
y=data['Perf']
X.keys()
data

# Use scikitlearn to do linear regression

my_rand_state = 42 # Random state may change test error.
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,
                                            random_state=my_rand_state)

#Standard Linear Regressor
linearRegressor = LinearRegression()
linearRegressor.fit(X_train, y_train)

print('train error:', 
      mean_squared_error(y_train, linearRegressor.predict(X_train)))

print('test error:', 
      mean_squared_error(y_test, linearRegressor.predict(X_test)))

#Standard Linear Regressor With PowerTransformer Pipeline
pl_linear = Pipeline([
    ('PowerTransformer', PowerTransformer()),
    ('linear', LinearRegression())
])

pl_linear.fit(X_train, y_train)

print('train error:', mean_squared_error(y_train, pl_linear.predict(X_train)))
print('test error:', mean_squared_error(y_test, pl_linear.predict(X_test)))

neigh = KNeighborsRegressor(n_neighbors=15).fit(X_train, y_train)
print('train error:', mean_squared_error(y_train, neigh.predict(X_train)))
print('test error:', mean_squared_error(y_test, neigh.predict(X_test)))

pl_neigh = Pipeline([
    ('PowerTransformer', PowerTransformer()),
    ('neigh', KNeighborsRegressor(n_neighbors=15))
]).fit(X_train, y_train)
print('train error:', mean_squared_error(y_train, pl_neigh.predict(X_train)))
print('test error:', mean_squared_error(y_test, pl_neigh.predict(X_test)))
