"""
Converted from: Chapter 4 - 10 - Random Forest Regression.ipynb

This script contains code extracted from a Jupyter notebook.
Code has been organized for modular execution.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""
Random Forest for Regression
"""

# API Doc:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor

# Compare Visually with Decision Tree

# We see that the random forest regressor allows us to set a high tree depth (because it is averaging across a forest of trees)

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [7.0, 4.5]
plt.rcParams['figure.dpi'] = 150

data = pd.read_csv('LinRegData.csv') # Read in our data, X and Y are series
x=data['X'].values.reshape(-1,1) # scikitlearn wants arguments as numpy arrays of this shape
y=data['Y'].values.reshape(-1,1)

treeRegressor = DecisionTreeRegressor(max_depth=4)
RFRegressor = RandomForestRegressor(max_depth=4, n_estimators=100)

treeRegressor.fit(x, y)
RFRegressor.fit(x, y)

x1 = np.linspace(0, 4, 50).reshape(-1, 1) # Make the predictions over the entire of X to see the tree splits
y_pred_tree = treeRegressor.predict(x1)
y_pred_forest = RFRegressor.predict(x1)

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

# Compare Visually with ExtraTrees

# Get data
data = pd.read_csv('LinRegData.csv') # Read in our data
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
treeRegressor = DecisionTreeRegressor(max_depth=4)

RFRegressor = RandomForestRegressor(max_depth=4, n_estimators=100)

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

# Compare with real data

data = pd.read_csv('stock_data_performance_fundamentals_300.csv', index_col=0) # Read in our data

X=data.drop(columns='Perf')
Y=data['Perf']
X.keys()

# First Tree Regression

# Use scikitlearn to do tree regression

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

treeRegressor = DecisionTreeRegressor(max_depth=1)

treeRegressor.fit(X_train, y_train)
y_pred = treeRegressor.predict(X_train)

print('train error', mean_squared_error(y_train, y_pred))
print('test error', mean_squared_error(y_test, treeRegressor.predict(X_test)))

# Next random forest regression

# Use scikitlearn to do tree regression

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

RFRegressor = RandomForestRegressor()

RFRegressor.fit(X_train, y_train)
y_pred = RFRegressor.predict(X_train)

print('train error', mean_squared_error(y_train, y_pred))
print('test error', mean_squared_error(y_test, RFRegressor.predict(X_test)))

# Plot learning curves for the two
