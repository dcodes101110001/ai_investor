"""
Converted from: Chapter 4 - 9 - KNN.ipynb

This script contains code extracted from a Jupyter notebook.
Code has been organized for modular execution.
"""

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

"""
K-Nearest Neighbours Example, 2 Features
"""
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

# hand-make example data from worked example
X = pd.DataFrame([[1,8],
                  [9,5],
                  [5,7],
                  [7,3],
                  [3,6],
                  [1,2],
                  [4,5]], columns=['X1','X2'])
y = pd.DataFrame([9,8,2,6,2,4,1], columns=['y'])

# Make KNN regressor
KNN = KNeighborsRegressor(n_neighbors=4).fit(X, y)

# See prediction
print('K-Nearest Neighbours prediction value is:', 
      KNN.predict([[5,4]]).ravel())

# hand-make example data from worked example
X = pd.DataFrame([[1,8],
                  [9,5],
                  [5,7],
                  [7,3],
                  [3,6],
                  [1,2],
                  [4,5]], columns=['X1','X2'])
y = pd.DataFrame([9,8,2,6,2,4,1], columns=['y'])
print(X)
print('\n')
print(y)
print('\n')

KNN = KNeighborsRegressor(n_neighbors=4).fit(X, y)
print('K-Nearest Neighbours prediction value is:', KNN.predict([[5,4]])[0][0]) # Answer is list of list, so to get number[0][0]

# | Row Number | y | X1 | X2 |
# | :- | :-: | :-: | :-: |
# | 0 | 9 | 1 | 8 |
# | 0 | 8 | 9 | 5 |
# | 0 | 2 | 5 | 7 |
# | 0 | 6 | 7 | 3 |
# | 0 | 2 | 3 | 6 |
# | 0 | 4 | 1 | 2 |
# | 0 | 1 | 4 | 5 |
#
#
#
#
# | Row Number | y | X1 | X2 | 
# | :- | :-: | :-: | :-: |
# | 0 | 9 | 1 | 8 |
# | 0 | 8 | 9 | 5 |
# | 0 | 2 | 5 | 7 |
# | 0 | 6 | 7 | 3 |
# | 0 | 2 | 3 | 6 |
# | 0 | 4 | 1 | 2 |
# | 0 | 1 | 4 | 5 |

# Try the Manhattan Distance Instead of Euclidian
# The p value changes the distance. Ways to calculate distance have been generalised in Minkowski distance.
# https://en.wikipedia.org/wiki/Minkowski_distance
# p=1 is the Manhattan distance, p=2 is Euclidian distance

KNN1 = KNeighborsRegressor(n_neighbors=2, p=1).fit(X, y) # Manhattan Distance
KNN2 = KNeighborsRegressor(n_neighbors=2, p=2).fit(X, y) # Euclidian Distance
print('K-Nearest Neighbours prediction value when p=1 is:', KNN1.predict([[5,4]])[0][0])
print('K-Nearest Neighbours prediction value when p=2 is:', KNN2.predict([[5,4]])[0][0])
