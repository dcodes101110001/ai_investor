#!/usr/bin/env python3
# Converted from: Chapter 4 - 10 - Random Forest Regression.ipynb


# ==============================================================================
# Cell 1
# ==============================================================================
# # Random Forest for Regression


# ==============================================================================
# Cell 2
# ==============================================================================

# API Doc:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor


# ==============================================================================
# Cell 3
# ==============================================================================
# ### Compare Visually with Decision Tree


# ==============================================================================
# Cell 4
# ==============================================================================
# We see that the random forest regressor allows us to set a high tree depth (because it is averaging across a forest of trees)


# ==============================================================================
# Cell 5
# ==============================================================================

import pandas as pd
import numpy as np


# ==============================================================================
# Cell 6
# ==============================================================================

# Set the plotting DPI settings to be a bit higher.
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [7.0, 4.5]
plt.rcParams['figure.dpi'] = 150


# ==============================================================================
# Cell 7
# ==============================================================================

data = pd.read_csv('stock_data\LinRegData.csv') # Read in our data, X and Y are series
x=data['X'].values.reshape(-1,1) # scikitlearn wants arguments as numpy arrays of this shape
y=data['Y'].values.reshape(-1,1)


# ==============================================================================
# Cell 8
# ==============================================================================

from sklearn.tree import DecisionTreeRegressor
treeRegressor = DecisionTreeRegressor(max_depth=4)
from sklearn.ensemble import RandomForestRegressor
RFRegressor = RandomForestRegressor(max_depth=4, n_estimators=100)


# ==============================================================================
# Cell 9
# ==============================================================================

treeRegressor.fit(x, y)
RFRegressor.fit(x, y)


# ==============================================================================
# Cell 10
# ==============================================================================

x1 = np.linspace(0, 4, 50).reshape(-1, 1) # Make the predictions over the entire of X to see the tree splits
y_pred_tree = treeRegressor.predict(x1)
y_pred_forest = RFRegressor.predict(x1)


# ==============================================================================
# Cell 11
# ==============================================================================

plt.scatter(x, y, color='orange') # scatter plot, learning data
plt.plot(x1, y_pred_tree, 'green') # plot of regression prediction on top (green line)
plt.plot(x1, y_pred_forest, 'blue') # plot of regression prediction on top (blue line)

plt.xlabel('X') # formatting
plt.ylabel('Y')
plt.grid()
plt.ylim([0,6]);
plt.xlim([0,4]);
plt.legend(['Single Decision Tree', 'Random Forest (100 Trees)'])
plt.title('Regression Max. Tree Depth=4');


# ==============================================================================
# Cell 12
# ==============================================================================
# ### Compare Visually with ExtraTrees


# ==============================================================================
# Cell 13
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # scikitlearn uses numpy arrays

# Get data
data = pd.read_csv('stock_data\LinRegData.csv') # Read in our data
data.head() # See the first few rows to understand the data

# Data is X and Y coordinates
X = data['X']
Y = data['Y']
plt.scatter(X, Y) # Plot 
plt.ylim([0,6])
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')

# Use scikitlearn to do regression
from sklearn.tree import DecisionTreeRegressor
treeRegressor = DecisionTreeRegressor(max_depth=4)

from sklearn.ensemble import RandomForestRegressor
RFRegressor = RandomForestRegressor(max_depth=4, n_estimators=100)

from sklearn.ensemble import ExtraTreesRegressor
ETRegressor = ExtraTreesRegressor(max_depth=4, n_estimators=100)

x=X.values.reshape(-1,1) # object wants arguments as numpy objects
y=Y.values.reshape(-1,1)
treeRegressor.fit(x, y)
RFRegressor.fit(x, y)
ETRegressor.fit(x, y)

x1 = np.linspace(0, 4, 50).reshape(-1, 1) # Make the predictions over the entire of X to see the tree splits

y_pred_tree = treeRegressor.predict(x1)
y_pred_forest = RFRegressor.predict(x1)
y_pred_extree = ETRegressor.predict(x1)

plt.scatter(X, Y) # scatter plot, learning data
plt.plot(x1, y_pred_tree, 'green') # plot of regression prediction on top (red line)
plt.plot(x1, y_pred_forest, 'blue') # plot of regression prediction on top (red line)
plt.plot(x1, y_pred_extree, 'red') # plot of regression prediction on top (red line)

#Plot formatting.
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.ylim([0,6]);
plt.legend(['Single Decision Tree', 'Random Forest (100 Trees)', 'Extra Trees'])
plt.title('Max. Tree Depth=4')
plt.grid()


# ==============================================================================
# Cell 14
# ==============================================================================
# ### Compare with real data


# ==============================================================================
# Cell 15
# ==============================================================================

data = pd.read_csv('stock_data\stock_data_performance_fundamentals_300.csv', index_col=0) # Read in our data


# ==============================================================================
# Cell 16
# ==============================================================================

X=data.drop(columns='Perf')
Y=data['Perf']
X.keys()


# ==============================================================================
# Cell 17
# ==============================================================================
# ### First Tree Regression


# ==============================================================================
# Cell 18
# ==============================================================================

# Use scikitlearn to do tree regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

treeRegressor = DecisionTreeRegressor(max_depth=1)

treeRegressor.fit(X_train, y_train)
y_pred = treeRegressor.predict(X_train)

print('train error', mean_squared_error(y_train, y_pred))
print('test error', mean_squared_error(y_test, treeRegressor.predict(X_test)))


# ==============================================================================
# Cell 19
# ==============================================================================
# ### Next random forest regression


# ==============================================================================
# Cell 20
# ==============================================================================

# Use scikitlearn to do tree regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

RFRegressor = RandomForestRegressor()

RFRegressor.fit(X_train, y_train)
y_pred = RFRegressor.predict(X_train)

print('train error', mean_squared_error(y_train, y_pred))
print('test error', mean_squared_error(y_test, RFRegressor.predict(X_test)))


# ==============================================================================
# Cell 21
# ==============================================================================
# ### Plot learning curves for the two
