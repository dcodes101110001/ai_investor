#!/usr/bin/env python3
# Converted from: Chapter 4 - 6 - Exercise 10 - Linear Regression.ipynb


# ==============================================================================
# Cell 1
# ==============================================================================
# # Linear Regression


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
# ## Single Feature Variable with Scikit-Learn Linear Regression


# ==============================================================================
# Cell 4
# ==============================================================================

# API Doc:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression


# ==============================================================================
# Cell 5
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # scikitlearn uses numpy arrays

data = pd.read_csv('LinRegData.csv') # Read in our data
data.head() # See the first few rows to understand the data


# ==============================================================================
# Cell 6
# ==============================================================================

# Data is X and Y coordinates
X_ = data['X']
y_ = data['Y']
plt.scatter(X_, y_) # Plot 
plt.ylim([0,6])
plt.grid()
plt.xlabel('X')
plt.ylabel('y');


# ==============================================================================
# Cell 7
# ==============================================================================

# Use scikitlearn to do linear regression
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()

x=X_.values.reshape(-1,1) # LinearRegressor object wants arguments as numpy objects
y=y_.values.reshape(-1,1)
linearRegressor.fit(x, y)
y_pred = linearRegressor.predict(x)
plt.scatter(X_, y_) # scatter plot, learning data
plt.plot(x, y_pred, 'red') # plot of linear regression prediction on top (red line)

#Plot formatting.
plt.xlabel('X')
plt.ylabel('y')
plt.grid()
plt.ylim([0,6]);
plt.legend(['Prediction'])


# ==============================================================================
# Cell 8
# ==============================================================================

# Just try a single number
linearRegressor.predict([[3]])


# ==============================================================================
# Cell 9
# ==============================================================================
# Root Mean Squared Error:
#
# $MSE\ =\ \frac{1}{n}\sum_{i=1}^{n}\left({\hat{Y}}_i - Y_i\right)^2$


# ==============================================================================
# Cell 10
# ==============================================================================

from sklearn.metrics import mean_squared_error # Sklearn library has this as a func for us.
mean_squared_error(y, y_pred)


# ==============================================================================
# Cell 11
# ==============================================================================
# ## Linear Regression Under the Hood


# ==============================================================================
# Cell 12
# ==============================================================================

# Maky your own simple linear regression Gradient Descent
eta = 0.05  # Learning Rate 0.01
iterations = 100  # The number of iterations to perform 1e3
n = len(X_) # Number of elements 
beta0, beta1 = 0, 0 # start with random numbers

# Performing Gradient Descent
for i in range(iterations): 
    beta0 = beta0 - eta * (2/n) * sum(beta0 + beta1 * X_ - y_)
    beta1 = beta1 - eta * (2/n) * sum((beta0 + beta1 * X_ - y_) * X_)

print('Final values of Beta0 and Beta1 are:', beta0, beta1)

y_pred = beta0 + beta1 * X_ # Do the prediction


# ==============================================================================
# Cell 13
# ==============================================================================

# Plotting
plt.scatter(X_, y) 
plt.plot(X_, y_pred, color='red')

# Formatting
plt.ylim([0,6])
plt.grid()
plt.xlabel('X')
plt.ylabel('y')
plt.legend(['Prediction']);


# ==============================================================================
# Cell 14
# ==============================================================================

mean_squared_error(y, y_pred)


# ==============================================================================
# Cell 15
# ==============================================================================
# # Exercise 9 - Custom Linear Regression with 2 Feature Variables
# Here we will take in 2 columns of stock fundamental data, as well as the past stock return, and try and predict returns
#
# Some of the code is missing. Try and fix the code, replacing the ??? text.


# ==============================================================================
# Cell 16
# ==============================================================================

data = pd.???_csv('Exercise_10_StockReturnData.csv') # Read in our data
data.head???) # Take a first look at the data with head


# ==============================================================================
# Cell 17
# ==============================================================================

data.describe() # data seems OK


# ==============================================================================
# Cell 18
# ==============================================================================

data.hist(); # histograms seem OK


# ==============================================================================
# Cell 19
# ==============================================================================

y = data['Stock Performance'] # Split data into X and Y
X = ???.drop(columns='Stock Performance')


# ==============================================================================
# Cell 20
# ==============================================================================

# Plotting scatter plot to see if anything obvious can be seen.
import matplotlib.cm as cm
colors = cm.rainbow(np.linspace(0, 1, len(y)))
plt.scatter(X['P/E'], X['RoE'], color=colors) # Plot
plt.xlabel('Price/Earnings')
plt.ylabel('Return on Equity')
plt.grid()
plt.xlim([-30,???])
plt.???lim([???,1]);
# Nothing easily to identify visually.


# ==============================================================================
# Cell 21
# ==============================================================================

from sklearn.model_selection import train_test_split # Want to split test and train data for Machine Learning
X_train, X_test, y_train, y_test = train_te???(X, ???, test_size=0.2)


# ==============================================================================
# Cell 22
# ==============================================================================

# Simple linear regression Gradient Descent
eta = 0.0002  # Learning Rate 0.0001 seems OK.
iterations = ???  # The number of iterations to perform, 1000 seems OK.
n = len(X_train) # Number of elements 
beta0, beta1, ??? = 0, 0, 0 # start with random numbers

# Manually setting X1 and X2. Could put regression into a function if you wish.
X1 = X_train['P/E']
X2 = X_train[???]

# Performing Gradient Descent to find Beta values.
for i in range(iterations): 
    beta0 = beta0 - eta * (2/n) * sum(beta0 + beta1*X1 + beta2*X2 - y_train)
    beta1 = beta1 - ??? * (2/n) * sum((beta0 + beta1*??? + beta2*X2 - ???) * X1)
    beta2 = beta2 - eta * (2/n) * sum((beta0 + beta1*X1 + beta2*??? - y_train) * X2)

y_pred = beta0 + beta1 * X1 + beta2 * X2 # Do the prediction
print('Final values of Beta0, Beta1 and Beta2 are:', beta0, beta1, beta2) # Make sure not crazy numbers


# ==============================================================================
# Cell 23
# ==============================================================================
# Model equation is:
#
# $\hat{y} = \beta_{0}+ \beta_{1}X_{1}+ \beta_{2}X_{2}$
#
# See book formulas for how $\beta_{0}, \beta_{1}, \beta_{2}$ values are found (code above)


# ==============================================================================
# Cell 24
# ==============================================================================

from sklearn.metrics import mean_squared_error
print('train error', mean_squared_error(y_train, y_pred))
print('test error', mean_squared_error(y_test, beta0 + beta1 * X_test['P/E'] + beta2 * X_test['RoE']))


# ==============================================================================
# Cell 25
# ==============================================================================

def predictReturn(PE=X['P/E'].mean(), 
                  RoE=X['RoE'].mean(), 
                  beta0=beta0, beta1=beta1, beta2=beta2):
    '''
    Prediction from 2 variable linear regression for P/E and RoE predicting stock return.
    Default takes the mean P/E or RoE (return on equity) values.
    Model fitted parameters beta0, beta1, beta2 are needed.
    '''
    stock_return_pred = beta0 + beta1*PE + beta2*RoE
    return stock_return_pred


# ==============================================================================
# Cell 26
# ==============================================================================

# The fit seems to work. Let's try a common-sense check to see if the algorithm is working.
# A stock with a ridiculously low P/E with a ridiculously high RoE should give us spectacular returns.
# Try P/E of 2 and a RoE of 100%
stock_return_pred = predictReturn(2, 1)
print('Predicted Stock Return P/E of 2 and RoE of 100% is:\n', 
      round(stock_return_pred*100,2),
      '%')


# ==============================================================================
# Cell 27
# ==============================================================================

# 7% as a prediction kind of sucks, but it is at least positive.
# Try P/E of 100 and a RoE of 10%, this kind of stock should do badly.
stock_return_pred = predictReturn(???, 0.1, beta0, beta1, beta2)
print('Predicted Stock Return P/E of 100 and RoE of 10% is:\n', 
      round(stock_return_pred*100,2),
      '%')


# ==============================================================================
# Cell 28
# ==============================================================================

'''
2% return is quite low for a stock return.
Bear in mind that we have less than 100 rows of data to work with. 

Our regression algorithm generally predicts things in the right direction, 
we expect a low P/E stock with high Return on Equity to perform better.

Also bear in mind that only beta0, 1 and 2 are changing, 
by their nature they will not capture many relationships in the data.

We'll use learning curves with Scikit-Learn linear regressors next.
''';


# ==============================================================================
# Cell 29
# ==============================================================================
# # Linear Regression Regularisation


# ==============================================================================
# Cell 30
# ==============================================================================

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
data=pd.read_csv('Exercise_10_stock_data_performance_fundamentals.csv', 
                 index_col=0)
data


# ==============================================================================
# Cell 31
# ==============================================================================

#data = pd.read_csv('Exercise_9_stock_data_performance_fundamentals.csv', index_col=0) # Read in our data
#from pandas.plotting import scatter_matrix # If want scatter matrix
#scatter_matrix(data, alpha=0.2, figsize=(10, 10))


# ==============================================================================
# Cell 32
# ==============================================================================

X=data.drop(columns='Perf')
y=data['Perf']
X.keys()


# ==============================================================================
# Cell 33
# ==============================================================================

# Use scikitlearn to do linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

linearRegressor = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=1234)

linearRegressor.fit(X_train, y_train)
y_pred = linearRegressor.predict(X_train)

print('train error:', 
      mean_squared_error(y_train, y_pred))

print('test error:', 
      mean_squared_error(y_test, linearRegressor.predict(X_test)))


# ==============================================================================
# Cell 34
# ==============================================================================

# Learning curve for any Regressor
def learningCurve(myModel, X, y, randomState):
        
    testErr, trainErr, trainSize = [],[],[] # lists as output
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=randomState)
    trainSize = range(1, len(X))
    for i in trainSize:
        myModel.fit(X_train[:i], y_train[:i])
        y_pred = myModel.predict(X_train[:i])
        
        trainErr.append(mean_squared_error(y_train[:i], 
                                           y_pred))
        
        testErr.append(mean_squared_error(y_test, 
                                          myModel.predict(X_test)))
        
    return np.sqrt(testErr), np.sqrt(trainErr), trainSize



from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


# ==============================================================================
# Cell 35
# ==============================================================================

# Plotting the learning curve shows some interesting things.
randomState=123 # randomstate try 123 and 1234
testErr, trainErr, trainSize = learningCurve(linearRegressor, X, y, randomState) 
plt.plot(trainSize, trainErr, trainSize, testErr) # plot
plt.legend(['Error vs. Training Set', 'Error vs. Testing Set']) # Formatting
plt.grid()
plt.xlabel('Rows in training set')
plt.ylabel('RMSE')
plt.ylim([0,6])
plt.xlim([0,80])
plt.title('Random State of {}'.format(randomState));


# ==============================================================================
# Cell 36
# ==============================================================================

from sklearn.linear_model import Ridge
ridgeRegressor = Ridge()
ridgeRegressor.fit(X_train, y_train)
y_pred = ridgeRegressor.predict(X_train)

print('train error', mean_squared_error(y_train, y_pred))
print('test error', mean_squared_error(y_test, ridgeRegressor.predict(X_test)))


# ==============================================================================
# Cell 37
# ==============================================================================

from sklearn.linear_model import Lasso
lassoRegressor = Lasso(alpha=0.1)
lassoRegressor.fit(X_train, y_train)
y_pred = lassoRegressor.predict(X_train)

print('train error', mean_squared_error(y_train, y_pred))
print('test error', mean_squared_error(y_test, lassoRegressor.predict(X_test)))


# ==============================================================================
# Cell 38
# ==============================================================================

from sklearn.linear_model import ElasticNet
eNetRegressor = ElasticNet(alpha=0.1, l1_ratio=0.5)
eNetRegressor.fit(X_train, y_train)
y_pred = eNetRegressor.predict(X_train)

print('train error', mean_squared_error(y_train, y_pred))
print('test error', mean_squared_error(y_test, eNetRegressor.predict(X_test)))


# ==============================================================================
# Cell 39
# ==============================================================================
# ##### Plot learning curves for regularised linear models


# ==============================================================================
# Cell 40
# ==============================================================================

# Plotting the learning curves for Linear Regressors
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

ridgeRegressor = Ridge()
lassoRegressor = Lasso(alpha=0.1)
eNetRegressor = ElasticNet(alpha=0.1, l1_ratio=0.5)

randomState=42 # randomstate
testErr1, trainErr1, trainSize1 = learningCurve(linearRegressor, 
                                                X, y, randomState)

testErr2, trainErr2, trainSize2 = learningCurve(ridgeRegressor, 
                                                X, y, randomState)

testErr3, trainErr3, trainSize3 = learningCurve(lassoRegressor, 
                                                X, y, randomState)

testErr4, trainErr4, trainSize4 = learningCurve(eNetRegressor, 
                                                X, y, randomState)


# ==============================================================================
# Cell 41
# ==============================================================================

fig, axs = plt.subplots(2, 2, figsize=(13,10))
axs[0, 0].plot(trainSize1, trainErr1, trainSize1, testErr1, 'tab:purple')
axs[0, 0].set_title("Linear Regressor")
axs[1, 0].plot(trainSize2, trainErr2, trainSize2, testErr2, 'tab:orange')
axs[1, 0].set_title("Ridge Regressor")
axs[0, 1].plot(trainSize3, trainErr3, trainSize3, testErr3, 'tab:green')
axs[0, 1].set_title("Lasso Regressor")
axs[1, 1].plot(trainSize4, trainErr4, trainSize4, testErr4, 'tab:red')
axs[1, 1].set_title("Elastic-Net Regressor")

for ax in axs.flat:
    ax.set(xlabel='Rows in training set', ylabel='RMSE', ylim=(0,4), xlim = (0,70))
    ax.grid()
    ax.legend(['Error vs. Training Set', 'Error vs. Testing Set'])
    
    # Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()


# ==============================================================================
# Cell 42
# ==============================================================================
# | Regressor | RMSE vs. Training Data | RMSE vs. Testing Data |
# | :- | :-: | :-: |
# | Linear | 0.129 | 3.894
# | Ridge | 0.133 | 0.752
# | Lasso | 0.146 | 0.116
# | Elastic-Net | 0.144 | 0.119


# ==============================================================================
# Cell 43
# ==============================================================================
# # Try the Scikitlearn Learning_Curve Function
# Over multiple runs to see the statistics.


# ==============================================================================
# Cell 44
# ==============================================================================

from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

def plotMyLearningCurve(regressor):
    '''
    test
    '''
    train_sizes, train_scores, test_scores = learning_curve(regressor, 
                                                             X, y, 
                                                             train_sizes=np.linspace(.1, 1.0, 10), 
                                                             cv=5,
                                                             scoring=make_scorer(mean_squared_error))
    # Use RMSE
    train_scores=np.sqrt(train_scores)
    test_scores=np.sqrt(test_scores)

    # Means and std. devs.
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    #plot
    plt.plot(train_sizes, train_scores.mean(axis=1), '-x')
    plt.fill_between(train_sizes,
                     train_scores.mean(axis=1)-train_scores.std(axis=1),
                     train_scores.mean(axis=1)+train_scores.std(axis=1),
                     alpha=0.2)
    plt.plot(train_sizes, test_scores.mean(axis=1), '-o')
    plt.fill_between(train_sizes,
                     test_scores.mean(axis=1)-test_scores.std(axis=1),
                     test_scores.mean(axis=1)+test_scores.std(axis=1),
                     alpha=0.2)
    
    
    
    plt.legend(['Train', 'Test'])
    plt.ylabel('RMSE')
    plt.xlabel('Rows in Training Set')
    plt.ylim([-0.1,3])
    plt.grid()
    
    pass


# ==============================================================================
# Cell 45
# ==============================================================================

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plotMyLearningCurve(linearRegressor);
plt.title('Linear Regressor', fontsize=15)
plt.subplot(2,2,2)
plotMyLearningCurve(lassoRegressor);
plt.title('Lasso Regressor', fontsize=15)
plt.subplot(2,2,3)
plotMyLearningCurve(ridgeRegressor);
plt.title('Ridge Regressor', fontsize=15)
plt.subplot(2,2,4)
plotMyLearningCurve(eNetRegressor);
plt.title('Elastic-Net Regressor', fontsize=15);


# ==============================================================================
# Cell 46
# ==============================================================================
# # Try with Shufflesplit to see how variable it can be
# Instead of K-fold splitting.
# The results aren't good, but we know that the regressor is 'learning', also this is financial data, it's not going to be easy to find a relationship.


# ==============================================================================
# Cell 47
# ==============================================================================

from sklearn.model_selection import ShuffleSplit

train_sizes, train_scores, test_scores = learning_curve(eNetRegressor, 
                                                         X, y, 
                                                         train_sizes=np.linspace(.2, 1.0, 10), 
                                                         cv=ShuffleSplit(n_splits=30, test_size=0.2, random_state=0),
                                                         scoring=make_scorer(mean_squared_error))

train_scores=np.sqrt(train_scores)
test_scores=np.sqrt(test_scores)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'r-', train_sizes, test_scores_mean, 'b-')
plt.plot(train_sizes, train_scores_mean+train_scores_std,'r-.')
plt.plot(train_sizes, train_scores_mean-train_scores_std,'r-.')
plt.plot(train_sizes, test_scores_mean+test_scores_std,'b-.')
plt.plot(train_sizes, test_scores_mean-test_scores_std,'b-.')
plt.legend(['Train', 'Test'])
plt.ylabel('RMSE')
plt.xlabel('Rows in Training Set')
plt.grid()


# ==============================================================================
# Cell 48
# ==============================================================================
# # Check Stock Return Predictions


# ==============================================================================
# Cell 49
# ==============================================================================

eNetRegressor = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X,y)


# ==============================================================================
# Cell 50
# ==============================================================================

eNetRegressor.predict(X)
