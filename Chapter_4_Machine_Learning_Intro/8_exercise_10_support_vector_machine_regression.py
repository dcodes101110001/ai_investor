"""
Converted from: Chapter 4 - 8 - Exercise 10 - Support Vector Machine Regression.ipynb

This script contains code extracted from a Jupyter notebook.
Code has been organized for modular execution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.svm import SVR

"""
Support Vector Machine Regression
"""

# https://scikit-learn.org/stable/modules/svm.html#svm-regression
#
# $\begin{align}\begin{aligned}\min_ {w, b, \zeta, \zeta^*} \frac{1}{2} w^T w + C \sum_{i=1}^{n} (\zeta_i + \zeta_i^*)\\\begin{split}\textrm {subject to } & y_i - w^T \phi (x_i) - b \leq \varepsilon + \zeta_i,\\
#                       & w^T \phi (x_i) + b - y_i \leq \varepsilon + \zeta_i^*,\\
#                       & \zeta_i, \zeta_i^* \geq 0, i=1, ..., n\end{split}\end{aligned}\end{align}$

"""
Exercise 10 – Support Vector Regression with Scikit-Learn
"""

# Here an example dataset is fabricated with Scikit-Learn
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=3)
plt.scatter(X,y)

# Using the linear SVR
svm_linreg = SVR(kernel='linear', C=100, gamma='auto').fit(X,y)
plt.scatter(X,y)
plt.plot(X, svm_linreg.predict(X))

#simple non-linear SVM to demonstrate radial basis function SVM (non-linear)

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=3)
y=0.1*(y+np.sin(2*X).reshape(1,-1)*42)
plt.scatter(X,y)

X, y=X.reshape(-1,1), y.reshape(-1,1)
svm_rbfreg = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.2).fit(X,y.ravel())
plt.scatter(X,y)
x_=np.arange(-3,2,0.1).reshape(-1,1)
plt.plot(x_, svm_rbfreg.predict(x_))
svm_linreg.fit(X,y.ravel())
plt.plot(x_, svm_linreg.predict(x_))
plt.legend(['RBF','Linear','Data'])
