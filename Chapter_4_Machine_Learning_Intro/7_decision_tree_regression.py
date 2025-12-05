"""
Converted from: Chapter 4 - 7 - Decision Tree Regression.ipynb

This script contains code extracted from a Jupyter notebook.
Code has been organized for modular execution.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # scikitlearn uses numpy arrays
from sklearn.tree import DecisionTreeRegressor

"""
Decision Trees for Regression
"""

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [7.0, 4.5]
plt.rcParams['figure.dpi'] = 150

# API doc
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

# Get data
data = pd.read_csv('LinRegData.csv') # Read in our data
data.head() # See the first few rows to understand the data

# Data is X and Y coordinates
X = data['X']
y = data['Y']
plt.scatter(X, y) # Plot 
plt.ylim([0,6])
plt.grid()
plt.xlabel('X')
plt.ylabel('y');

# Use scikitlearn to do regression
#treeRegressor = DecisionTreeRegressor()
treeRegressor = DecisionTreeRegressor(max_depth=1)


x=X.values.reshape(-1,1) # object wants arguments as numpy objects
y=y.values.reshape(-1,1)
treeRegressor.fit(x, y)

x1 = np.linspace(0, 4, 50).reshape(-1, 1) # Make the predictions over the entire of X to see the tree splits

y_pred = treeRegressor.predict(x1)
plt.scatter(X, y) # scatter plot, learning data
plt.plot(x1, y_pred, 'red') # plot of regression prediction on top (red line)

#Plot formatting.
plt.xlabel('X')
plt.ylabel('y')
plt.grid()
plt.ylim([0,6]);
plt.legend(['Prediction'])
plt.title('Max Tree Depth=1');
