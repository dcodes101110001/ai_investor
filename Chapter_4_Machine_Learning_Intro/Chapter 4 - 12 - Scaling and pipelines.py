#!/usr/bin/env python3
# Converted from: Chapter 4 - 12 - Scaling and pipelines.ipynb


# ==============================================================================
# Cell 1
# ==============================================================================

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from pandas.plotting import scatter_matrix


# ==============================================================================
# Cell 2
# ==============================================================================

# Set the plotting DPI settings to be a bit higher.
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [7.0, 4.5]
plt.rcParams['figure.dpi'] = 150


# ==============================================================================
# Cell 3
# ==============================================================================

data = pd.read_csv('stock_data_performance_fundamentals_300.csv', index_col=0)
x = data.drop(columns='Perf')
y = data['Perf']


# ==============================================================================
# Cell 4
# ==============================================================================
# # First Standard Scaler


# ==============================================================================
# Cell 5
# ==============================================================================

x.describe()


# ==============================================================================
# Cell 6
# ==============================================================================

myKey = x.keys()[2]
x[myKey].hist(bins=50)


# ==============================================================================
# Cell 7
# ==============================================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() # Create Standard Scaler Object
x_s = scaler.fit_transform(x) # Fit the scaler, returns a numpy array
x_s = pd.DataFrame(x_s, columns=x.keys())


# ==============================================================================
# Cell 8
# ==============================================================================

x_s.describe()


# ==============================================================================
# Cell 9
# ==============================================================================

x_s[myKey].hist(bins=50)


# ==============================================================================
# Cell 10
# ==============================================================================
# ### Side-by-side


# ==============================================================================
# Cell 11
# ==============================================================================

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


# ==============================================================================
# Cell 12
# ==============================================================================
# ### Sometimes need more than just transform


# ==============================================================================
# Cell 13
# ==============================================================================

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


# ==============================================================================
# Cell 14
# ==============================================================================
# # Now Power Transformer


# ==============================================================================
# Cell 15
# ==============================================================================

from sklearn.preprocessing import PowerTransformer
transformer = PowerTransformer()
x_t=transformer.fit_transform(x)


# ==============================================================================
# Cell 16
# ==============================================================================

x_t = pd.DataFrame(x_t)
x_t.columns = x.keys()
x_t.describe()


# ==============================================================================
# Cell 17
# ==============================================================================

x_t[myKey].hist(bins=100)


# ==============================================================================
# Cell 18
# ==============================================================================
# # Have a look at scaling and transforming


# ==============================================================================
# Cell 19
# ==============================================================================

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


# ==============================================================================
# Cell 20
# ==============================================================================
# # Pipeline Example
# To use a transformer with Scikit-Learn, it is easier to assemble your transformations into a pipeline.


# ==============================================================================
# Cell 21
# ==============================================================================

data = pd.read_csv('stock_data_performance_fundamentals_300.csv', index_col=0) # Read in our data
X=data.drop(columns='Perf')
y=data['Perf']
X.keys()
data


# ==============================================================================
# Cell 22
# ==============================================================================

# Use scikitlearn to do linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

my_rand_state = 42 # Random state may change test error.
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,
                                            random_state=my_rand_state)


# ==============================================================================
# Cell 23
# ==============================================================================

#Standard Linear Regressor
linearRegressor = LinearRegression()
linearRegressor.fit(X_train, y_train)

print('train error:', 
      mean_squared_error(y_train, linearRegressor.predict(X_train)))

print('test error:', 
      mean_squared_error(y_test, linearRegressor.predict(X_test)))


# ==============================================================================
# Cell 24
# ==============================================================================

#Standard Linear Regressor With PowerTransformer Pipeline
pl_linear = Pipeline([
    ('PowerTransformer', PowerTransformer()),
    ('linear', LinearRegression())
])

pl_linear.fit(X_train, y_train)

print('train error:', mean_squared_error(y_train, pl_linear.predict(X_train)))
print('test error:', mean_squared_error(y_test, pl_linear.predict(X_test)))


# ==============================================================================
# Cell 25
# ==============================================================================

from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=15).fit(X_train, y_train)
print('train error:', mean_squared_error(y_train, neigh.predict(X_train)))
print('test error:', mean_squared_error(y_test, neigh.predict(X_test)))


# ==============================================================================
# Cell 26
# ==============================================================================

pl_neigh = Pipeline([
    ('PowerTransformer', PowerTransformer()),
    ('neigh', KNeighborsRegressor(n_neighbors=15))
]).fit(X_train, y_train)
print('train error:', mean_squared_error(y_train, pl_neigh.predict(X_train)))
print('test error:', mean_squared_error(y_test, pl_neigh.predict(X_test)))
