#!/usr/bin/env python3
# Converted from: Chapter 4 - 11 - Gradient Boosting.ipynb


# ==============================================================================
# Cell 1
# ==============================================================================

import numpy as np
from matplotlib import pyplot as plt


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
# # Create Own Decision Tree Gradient Booster


# ==============================================================================
# Cell 4
# ==============================================================================

np.random.seed(42)
X = np.random.rand(200, 1) * 1.5
y = np.cos(X[:, 0]*3) + 0.1 * np.random.randn(200)


# ==============================================================================
# Cell 5
# ==============================================================================

from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=123)
tree_reg1.fit(X, y)

y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=123)
tree_reg2.fit(X, y2)

y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=123)
tree_reg3.fit(X, y3)

y4 = y3 - tree_reg3.predict(X)
tree_reg4 = DecisionTreeRegressor(max_depth=2, random_state=123)
tree_reg4.fit(X, y4)


# ==============================================================================
# Cell 6
# ==============================================================================

def plotRow1(x, y, x1, y1, limitx, limity, legendDat):
    plt.scatter(X, y, color='black', s=10);
    plt.plot(x1, y1 ,'red')
    plt.xlim(limitx)
    plt.ylim(limity)
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend([legendDat,'Training Set'])

def plotRow2(x, y, x1, y1, 
             limitx, limity, 
             ylab, legendDat, lincol, dotcol):
    
    plt.scatter(X, y, color=dotcol, s=10);
    plt.plot(x1, y1 , lincol)
    plt.xlim(limitx)
    plt.ylim(limity)
    plt.grid()
    plt.xlabel('X')
    plt.ylabel(ylab)
    plt.legend([legendDat,'Previous prediction - Training Set'])

x1 = np.linspace(0, 1.5, 500).reshape(-1,1)
plt.figure(figsize=(15,10))

limitx=[0,1.5]
limity=[-1.2, 1.4]

plt.subplot(231)
plotRow1(X, y, x1, 
         tree_reg1.predict(x1), 
         limitx, limity, 'h(x)=h1(x)')

plt.subplot(232)
plotRow1(X, y, x1, 
         tree_reg1.predict(x1)+tree_reg2.predict(x1), 
         limitx, limity, 'h(x)=h1(x)-h2(x)')

plt.subplot(233)
plotRow1(X, y, x1, 
         tree_reg1.predict(x1)+tree_reg2.predict(x1)+tree_reg3.predict(x1), 
         limitx, limity, 'h(x)=h1(x)-h2(x)-h3(x)')

plt.subplot(234)
plotRow2(X, y2, x1, 
         tree_reg2.predict(x1), [0,1.5], [-0.6,0.6], 
         'y-h1(x)', 'h2(x)', 'fuchsia', 'blue')

plt.subplot(235)
plotRow2(X, y3, x1, 
         tree_reg3.predict(x1), [0,1.5], [-0.6,0.6], 
         'y-h1(x)-h2(x)', 'h3(x)', 'fuchsia', 'blue')


# ==============================================================================
# Cell 7
# ==============================================================================
# # Use SKLearn Gradient Boosted Decision Tree


# ==============================================================================
# Cell 8
# ==============================================================================

# API Doc:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor


# ==============================================================================
# Cell 9
# ==============================================================================

from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, 
                                 n_estimators=3, 
                                 learning_rate=1.0, 
                                 random_state=42)
gbrt.fit(X, y)
plt.scatter(X, y, s=10);
plt.plot(x1, gbrt.predict(x1), 'r')
plt.grid()
