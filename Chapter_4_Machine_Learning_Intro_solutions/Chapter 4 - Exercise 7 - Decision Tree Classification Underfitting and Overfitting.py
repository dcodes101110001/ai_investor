#!/usr/bin/env python3
# Converted from: Chapter 4 - Exercise 7 - Decision Tree Classification Underfitting and Overfitting.ipynb


# ==============================================================================
# Cell 1
# ==============================================================================

import pandas as pd # Importing modules for use.
import numpy as np
import matplotlib.pyplot as plt # FOr plotting scatter plot


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
# # Exercise 7 on Decision Trees


# ==============================================================================
# Cell 4
# ==============================================================================

# Oh no, Toby has messed up the code again. Try and get it working,
# I think some code is missing where "?" is placed.


# ==============================================================================
# Cell 5
# ==============================================================================

'''
Here we want to plot the prediction accuracy with increasing decision tree depth
We want two accuracy lines plotted, one for accuracy vs. the testing set and
one for the accuracy vs the training set. 

We want to plot the tree depth from 1 to about 20.
We will loop through possible tree depth numbers, fitting our model with a different
tree depth with each loop, and appending the accuracy result to a list.

We will plot the list at the end.
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('Altman_Z_2D_Large.csv', index_col=0) # Load the .csv data

X = data[['EBIT/Total Assets','MktValEquity/Debt']]
Y = data['Bankrupt']

from sklearn.model_selection import train_test_split # need to import
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

scores_train, scores_test = [], []
level = 20
for i in range(1, level):
    tree_clf = DecisionTreeClassifier(max_depth=i) # create a DecisionTreeClassifier object first
    tree_clf.fit(X_train, y_train) # Fit the decision tree to our training data of X and Y.
    scores_train.append(accuracy_score(tree_clf.predict(X_train), y_train))
    scores_test.append(accuracy_score(tree_clf.predict(X_test), y_test))

plt.plot(range(1,level), scores_train, range(1,level), scores_test)
plt.legend(('Accuracy on Training Set','Accuracy on Testing Set'))
plt.grid()
plt.xlabel('Decision Tree Depth')
plt.ylabel('Prediction Accuracy')
