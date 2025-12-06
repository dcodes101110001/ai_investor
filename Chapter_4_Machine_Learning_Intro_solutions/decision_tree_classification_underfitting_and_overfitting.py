"""
Converted from: Chapter 4 - Exercise 7 - Decision Tree Classification Underfitting and Overfitting.ipynb

This script contains code extracted from a Jupyter notebook.
Code has been organized for modular execution.
"""

import pandas as pd # Importing modules for use.
import numpy as np
import matplotlib.pyplot as plt # FOr plotting scatter plot
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split # need to import

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [7.0, 4.5]
plt.rcParams['figure.dpi'] = 150

"""
Exercise 7 on Decision Trees
"""

# Oh no, Toby has messed up the code again. Try and get it working,
# I think some code is missing where "?" is placed.

# Here we want to plot the prediction accuracy with increasing decision tree depth
# We want two accuracy lines plotted, one for accuracy vs. the testing set and
# one for the accuracy vs the training set. 
#
# We want to plot the tree depth from 1 to about 20.
# We will loop through possible tree depth numbers, fitting our model with a different
# tree depth with each loop, and appending the accuracy result to a list.
#
# We will plot the list at the end.
