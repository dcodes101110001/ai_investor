#!/usr/bin/env python3
# Converted from: 3_Initial_Machine_Learning.ipynb


# ==============================================================================
# Cell 1
# ==============================================================================
# # Prediction of Annual Return from Fundamental Data and Market Cap
# Chapter 4 of the book: "Build Your Own AI Investor"
#
# For our investing AI to select stocks for investment it will need to be able to predict which stocks are likely to go up. WIth our X and y data we can train any of the machine learning algorithms to do this. We'll try using all of them and keep the ones that show promise.


# ==============================================================================
# Cell 2
# ==============================================================================

# Code from Book: Build Your Own AI Investor
# Damon Lee 2021
# Check out the performance on www.valueinvestingai.com
# Code uses data from the (presumably) nice people at https://simfin.com/. 
# Feel free to fork this code for others to see what can be done with it.


# ==============================================================================
# Cell 3
# ==============================================================================

from platform import python_version
print(python_version())


# ==============================================================================
# Cell 4
# ==============================================================================
# ### Imports and Getting Our Data


# ==============================================================================
# Cell 5
# ==============================================================================

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from pandas.plotting import scatter_matrix


# ==============================================================================
# Cell 6
# ==============================================================================

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [7.0, 4.5]
plt.rcParams['figure.dpi'] = 150


# ==============================================================================
# Cell 7
# ==============================================================================

def loadXandyAgain():
    '''
    Load X and y.
    Randomises rows.
    Returns X, y.
    '''
    # Read in data
    X=pd.read_csv("stock_data\\Annual_Stock_Price_Fundamentals_Ratios.csv",
                  index_col=0)
    y=pd.read_csv("stock_data\\Annual_Stock_Price_Performance_Percentage.csv",
                  index_col=0)
    y=y["Perf"] # We only need the % returns as target
    
    # randomize the rows
    X['y'] = y
    X = X.sample(frac=1.0, random_state=42) # randomize the rows
    y = X['y']
    X.drop(columns=['y'], inplace=True)
    
    return X, y


# ==============================================================================
# Cell 8
# ==============================================================================

X, y = loadXandyAgain()
y.mean() # Average stock return if we were picking at random.


# ==============================================================================
# Cell 9
# ==============================================================================
# # Linear Regression
# As a start try vanilla linear regression to get the ball rolling.
# We use the powertransformer in a pipeline with our linear regressor.


# ==============================================================================
# Cell 10
# ==============================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print('X training matrix dimensions: ', X_train.shape)
print('X testing matrix dimensions: ', X_test.shape)
print('y training matrix dimensions: ', y_train.shape)
print('y testing matrix dimensions: ', y_test.shape)


# ==============================================================================
# Cell 11
# ==============================================================================

# Try out linear regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

pl_linear = Pipeline([
    ('Power Transformer', PowerTransformer()),
    ('linear', LinearRegression())
    ])

pl_linear.fit(X_train, y_train)
y_pred = pl_linear.predict(X_test)

print('Train MSE: ',
      mean_squared_error(y_train, pl_linear.predict(X_train)))
print('Test MSE: ',
      mean_squared_error(y_test, y_pred))

#import pickle # To save the fitted model
#pickle.dump(pl_linear, open("stock_data\\pl_linear.p", "wb" ))


# ==============================================================================
# Cell 12
# ==============================================================================
# The Errors aren't that good, and they can vary a lot depending on train/test split. Let's try many runs and see if it the regressor is actually learning anything.


# ==============================================================================
# Cell 13
# ==============================================================================

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

sizesToTrain = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]   
train_sizes, train_scores, test_scores, fit_times, score_times = \
    learning_curve(pl_linear, X, y, cv=ShuffleSplit(n_splits=100,
                                                    test_size=0.2,
                                                    random_state=42),
                   scoring='neg_mean_squared_error',
                   n_jobs=4, train_sizes=sizesToTrain,
                   return_times=True)

results_df = pd.DataFrame(index=train_sizes) #Create a DataFrame of results
results_df['train_scores_mean'] = np.sqrt(-np.mean(train_scores, axis=1))
results_df['train_scores_std'] = np.std(np.sqrt(-train_scores), axis=1)
results_df['test_scores_mean'] = np.sqrt(-np.mean(test_scores, axis=1))
results_df['test_scores_std'] = np.std(np.sqrt(-test_scores), axis=1)
results_df['fit_times_mean'] = np.mean(fit_times, axis=1)
results_df['fit_times_std'] = np.std(fit_times, axis=1)


# ==============================================================================
# Cell 14
# ==============================================================================

results_df # see results


# ==============================================================================
# Cell 15
# ==============================================================================

results_df['train_scores_mean'].plot(style='-x')
results_df['test_scores_mean'].plot(style='-x')

plt.fill_between(results_df.index,\
                 results_df['train_scores_mean']-results_df['train_scores_std'],\
                 results_df['train_scores_mean']+results_df['train_scores_std'], alpha=0.2)
plt.fill_between(results_df.index,\
                 results_df['test_scores_mean']-results_df['test_scores_std'],\
                 results_df['test_scores_mean']+results_df['test_scores_std'], alpha=0.2)
plt.grid()
plt.legend(['Training CV RMSE Score','Test Set RMSE'])
plt.ylabel('RMSE')
plt.xlabel('Number of training rows')
plt.title('Linear Regression Learning Curve', fontsize=15)
#plt.ylim([0, 5]);


# ==============================================================================
# Cell 16
# ==============================================================================
# ### Linear Regression Prediction Analysis
# Let's see how good our predictions are in a bit more depth. Our AI will be depending on these predictions so we need to be sure stocks can be picked well in this chapter.


# ==============================================================================
# Cell 17
# ==============================================================================
# #### Plotting Function
# To get a better view of how good the predictions are(visually) without depending on mean squared error.


# ==============================================================================
# Cell 18
# ==============================================================================

# Output scatter plot and contour plot of density of points to see if prediciton matches reality
# Line of x=y is provided, perfect prediction would have all density on this line
# Also plot linear regression of the scatter

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plotDensityContourPredVsReal(model_name, x_plot, y_plot, ps):
    # Plotting scatter 
    plt.scatter(x_plot, y_plot, s=1)
    # Plotting linear regression
    # Swap X and Y fit because prediction is quite centered around one value.
    LinMod = LinearRegression().fit(y_plot.reshape(-1, 1), x_plot.reshape(-1, 1))
    xx=[[-5],[5]]
    yy=LinMod.predict(xx)
    plt.plot(yy,xx,'g')
    # Plot formatting
    plt.grid()
    plt.axhline(y=0, color='r', label='_nolegend_')
    plt.axvline(x=0, color='r', label='_nolegend_')
    plt.xlabel('Predicted Return')
    plt.ylabel('Actual Return')
    plt.plot([-100,100],[-100,100],'y--')
    plt.xlim([-ps,ps])
    plt.ylim([-ps,ps])
    plt.title('Predicted/Actual density plot for {}'.format(model_name))
    plt.legend(['Linear Fit Line','y=x Perfect Prediction Line','Prediction Points'])
    # Save Figure
    #plt.figure(figsize=(5,5))
    plt.savefig('result.png')


# ==============================================================================
# Cell 19
# ==============================================================================

def plotPredictionVsTestDataTopBottom(y_pred, y_test, l=100, plotLimits=[-1,7]):
    comparisonData=pd.DataFrame({'y_pred':y_pred, 
                                 'y_test':y_test}).sort_values(by='y_pred')

    end = comparisonData.shape[0]-1
    start = end - l
    select1 = list(range(0,l)) # 0 to N
    select2 = list(range(start,end)) # last N
    plt.figure(figsize=(14,7))

    plt.subplot(1,2,2)
    plt.plot(select2, comparisonData.iloc[-l:],'x')
    plt.ylim(plotLimits)
    plt.grid()
    plt.ylabel('Stock Annual Return');
    plt.xlabel('Data Row Number (rank lowest to highest predicted return)');
    plt.legend(['Stock Predicted Return',
                'Stock Actual Return', 
                'Mean of actual ']);

    plt.subplot(1,2,1)
    plt.plot(select1, comparisonData.iloc[:l],'x')
    plt.ylim(plotLimits)
    plt.grid()
    plt.ylabel('Stock Annual Return');
    plt.xlabel('Data Row Number (rank lowest to highest predicted return)');
    plt.legend(['Stock Predicted Return',
                'Stock Actual Return']);
    pass

plotPredictionVsTestDataTopBottom(y_pred, y_test, l=100)


# ==============================================================================
# Cell 20
# ==============================================================================

plt.figure(figsize=(6,6))
plotDensityContourPredVsReal('pl_linear', y_pred, y_test.to_numpy(), 2)


# ==============================================================================
# Cell 21
# ==============================================================================
# It doesn't look so good visually, but there seems to be some ability there. Let's take a closer look by taking our predictors top few stock return predictions and comparing them to reality.


# ==============================================================================
# Cell 22
# ==============================================================================

# See top 10 stocks and see how the values differ
# Put results in a DataFrame so we can sort it.
y_results = pd.DataFrame()
y_results['Actual Return'] = y_test
y_results['Predicted Return'] = y_pred

# Sort it by the prediced return.
y_results.sort_values(by='Predicted Return',
                      ascending=False,
                      inplace=True)
y_results.reset_index(drop=True,
                      inplace=True)


print('Predicted Returns:', list(np.round(y_results['Predicted Return'].iloc[:10],2)))
print('Actual Returns:', list(np.round(y_results['Actual Return'].iloc[:10],2)))
print('\nTop 10 Predicted Returns:', round(y_results['Predicted Return'].iloc[:10].mean(),2) , '%','\n')
print('Actual Top 10 Returns:', round(y_results['Actual Return'].iloc[:10].mean(),2) , '%','\n')


# ==============================================================================
# Cell 23
# ==============================================================================
# Let's try the bottom 10.


# ==============================================================================
# Cell 24
# ==============================================================================

# See bottom 10 stocks and see how the values differ
print('\nBottom 10 Predicted Returns:', round(y_results['Predicted Return'].iloc[-10:].mean(),2) , '%','\n')
print('Actual Bottom 10 Returns:', round(y_results['Actual Return'].iloc[-10:].mean(),2) , '%','\n')


# ==============================================================================
# Cell 25
# ==============================================================================
# There is *some* predictive ability here, it is definately worthwhile using linear regression in the backtest under greater scrutiny later. Don't worry about survivorship bias at this point, we can account for that later, besides these predicted returns are high enough to compensate for that if you look at bankruptcy statistics, and the kinds of stocks being chosen.
#
# Let's try the model with a few more train/test samplings by changing the random_state, and see if the predictive ability stays.


# ==============================================================================
# Cell 26
# ==============================================================================
# ### Try a few Linear Regression runs and see if the top/bottom 10 selections are any good.


# ==============================================================================
# Cell 27
# ==============================================================================

def observePredictionAbility(my_pipeline, X, y, returnSomething=False, verbose=True):
    '''
    For a given predictor pipeline.
    Create table of top10/bottom 10 averaged, 
    10 rows of 10 random_states.
    to give us a synthetic performance result.    
    Prints Top and Bottom stock picks
    
    The arguments returnSomething=False, verbose=True,
    will be used at the notebook end to get results.
    '''
    Top10PredRtrns, Top10ActRtrns=[], []
    Bottom10PredRtrns, Bottom10ActRtrns=[], []

    for i in range (0, 10): # Can try 100
        # Pipeline and train/test
        X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.1, random_state=42+i)
        my_pipeline.fit(X_train, y_train)
        y_pred = my_pipeline.predict(X_test)
        
        # Put results in a DataFrame so we can sort it.
        y_results = pd.DataFrame()
        y_results['Actual Return'] = y_test
        y_results['Predicted Return'] = y_pred
        
        # Sort it by the prediced return.
        y_results.sort_values(by='Predicted Return',
                              ascending=False,
                              inplace=True)
        y_results.reset_index(drop=True,
                              inplace=True)
        
        
         # See top 10 stocks and see how the values differ
        Top10PredRtrns.append(
            round(np.mean(y_results['Predicted Return'].iloc[:10])*100,
                  2))
        Top10ActRtrns.append(
            round(np.mean(y_results['Actual Return'].iloc[:10])*100,
                  2))
        
        # See bottom 10 stocks and see how the values differ
        Bottom10PredRtrns.append(
            round(np.mean(y_results['Predicted Return'].iloc[-10:])*100,
                  2))
        Bottom10ActRtrns.append(
            round(np.mean(y_results['Actual Return'].iloc[-10:])*100,
                  2))

    if verbose:
        print('Predicted Performance of Top 10 Return Portfolios:', 
              Top10PredRtrns)
        print('Actual Performance of Top 10 Return Portfolios:', 
              Top10ActRtrns,'\n')
        print('Predicted Performance of Bottom 10 Return Portfolios:', 
              Bottom10PredRtrns)
        print('Actual Performance of Bottom 10 Return Portfolios:', 
              Bottom10ActRtrns)
        print('--------------\n')
        
        print('Mean Predicted Std. Dev. of Top 10 Return Portfolios:',
              round(np.array(Top10PredRtrns).std(),2))
        print('Mean Actual Std. Dev. of Top 10 Return Portfolios:',
              round(np.array(Top10ActRtrns).std(),2))
        print('Mean Predicted Std. Dev. of Bottom 10 Return Portfolios:',
              round(np.array(Bottom10PredRtrns).std(),2))
        print('Mean Actual Std. Dev. of Bottom 10 Return Portfolios:',
              round(np.array(Bottom10ActRtrns).std(),2))
        print('--------------\n')
        
        #PERFORMANCE MEASURES HERE
        print(\
        '\033[4mMean Predicted Performance of Top 10 Return Portfolios:\033[0m',\
              round(np.mean(Top10PredRtrns), 2))
        print(\
        '\t\033[4mMean Actual Performance of Top 10 Return Portfolios:\033[0m',\
              round(np.mean(Top10ActRtrns), 2))
        print('Mean Predicted Performance of Bottom 10 Return Portfolios:',\
              round(np.mean(Bottom10PredRtrns), 2))
        print('\tMean Actual Performance of Bottom 10 Return Portfolios:',\
              round(np.mean(Bottom10ActRtrns), 2))
        print('--------------\n')
    
    if returnSomething:
        # Return the top10 and bottom 10 predicted stock return portfolios
        # (the actual performance)
        return Top10ActRtrns, Bottom10ActRtrns
    
    pass


# ==============================================================================
# Cell 28
# ==============================================================================

observePredictionAbility(pl_linear, X, y)


# ==============================================================================
# Cell 29
# ==============================================================================
# # Elastic Net Regression
# Let's see if some regularisation form linear regression will get us better results.


# ==============================================================================
# Cell 30
# ==============================================================================

# ElasticNet
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PowerTransformer

pl_ElasticNet = Pipeline([
    ('Power Transformer', PowerTransformer()),
    ('ElasticNet', ElasticNet())#l1_ratio=0.00001, alpha=0.001
])

pl_ElasticNet.fit(X_train, y_train)
y_pred = pl_ElasticNet.predict(X_test)
from sklearn.metrics import mean_squared_error
print('train mse: ', mean_squared_error(y_train, pl_ElasticNet.predict(X_train)))
print('test mse: ', mean_squared_error(y_test, pl_ElasticNet.predict(X_test)))

import pickle
pickle.dump(pl_ElasticNet, open("stock_data\\pl_ElasticNet.p", "wb" ))


# ==============================================================================
# Cell 31
# ==============================================================================

plt.figure(figsize=(5,5))
plotDensityContourPredVsReal('pl_ElasticNet', y_pred, y_test.to_numpy(),2)


# ==============================================================================
# Cell 32
# ==============================================================================
# ### Test some Hyperparameters to try and improve prediction


# ==============================================================================
# Cell 33
# ==============================================================================

# ElasticNet
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PowerTransformer

pl_ElasticNet = Pipeline([
    ('Power Transformer', PowerTransformer()),
    ('ElasticNet', ElasticNet(l1_ratio=0.00001))
])

pl_ElasticNet.fit(X_train, y_train)
y_pred_lowL1 = pl_ElasticNet.predict(X_test)
from sklearn.metrics import mean_squared_error
print('train mse: ', mean_squared_error(y_train, pl_ElasticNet.predict(X_train)))
print('test mse: ', mean_squared_error(y_test, pl_ElasticNet.predict(X_test)))

#import pickle
#pickle.dump(pl_ElasticNet, open("stock_data\\pl_ElasticNet.p", "wb" ))


# ==============================================================================
# Cell 34
# ==============================================================================

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plotDensityContourPredVsReal('pl_ElasticNet', y_pred, y_test.to_numpy(),2)
plt.title('Elasticnet Default Hyperparameters',fontsize=15)
plt.subplot(1,2,2)
plotDensityContourPredVsReal('pl_ElasticNet', y_pred_lowL1, y_test.to_numpy(),2)
plt.title('Elasticnet L1 Ratio=0.00001',fontsize=15)


# ==============================================================================
# Cell 35
# ==============================================================================

def PrintTopAndBottom10Predictions(y_test, y_pred):
    '''
    See top 10 stocks and see how the values differ.
    Returns nothing.
    '''
    
    # Put results in a DataFrame so we can sort it.
    y_results = pd.DataFrame()
    y_results['Actual Return'] = y_test
    y_results['Predicted Return'] = y_pred

    # Sort it by the prediced return.
    y_results.sort_values(by='Predicted Return',
                          ascending=False,
                          inplace=True)
    y_results.reset_index(drop=True,
                          inplace=True)


    #print('Predicted Returns:', list(np.round(y_results['Predicted Return'].iloc[:10],2)))
    #print('Actual Returns:', list(np.round(y_results['Actual Return'].iloc[:10],2)))
    #Top 10
    print('\nTop 10 Predicted Returns:', round(y_results['Predicted Return'].iloc[:10].mean(),2) , '%','\n')
    print('Actual Top 10 Returns:', round(y_results['Actual Return'].iloc[:10].mean(),2) , '%','\n')
    #Bottom 10
    print('\nBottom 10 Predicted Returns:', round(y_results['Predicted Return'].iloc[-10:].mean(),2) , '%','\n')
    print('Actual Bottom 10 Returns:', round(y_results['Actual Return'].iloc[-10:].mean(),2) , '%','\n')
    
    pass


# ==============================================================================
# Cell 36
# ==============================================================================

PrintTopAndBottom10Predictions(y_test, y_pred)


# ==============================================================================
# Cell 37
# ==============================================================================
# ### See if the results are repeatable.


# ==============================================================================
# Cell 38
# ==============================================================================

observePredictionAbility(pl_ElasticNet, X, y)


# ==============================================================================
# Cell 39
# ==============================================================================
# # K Nearest Neighbours Regression


# ==============================================================================
# Cell 40
# ==============================================================================

# Read in data and do train/test split.
X, y = loadXandyAgain()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print('X training matrix dimensions: ', X_train.shape)
print('X testing matrix dimensions: ', X_test.shape)
print('y training matrix dimensions: ', y_train.shape)
print('y testing matrix dimensions: ', y_test.shape)


# ==============================================================================
# Cell 41
# ==============================================================================

# KNeighbors regressor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PowerTransformer
pl_KNeighbors = Pipeline([
    ('Power Transformer', PowerTransformer()),
    ('KNeighborsRegressor', KNeighborsRegressor(n_neighbors=40))
])

pl_KNeighbors.fit(X_train, y_train)
y_pred = pl_KNeighbors.predict(X_test)
from sklearn.metrics import mean_squared_error
print('train mse: ', mean_squared_error(y_train, pl_KNeighbors.predict(X_train)))
print('test mse: ', mean_squared_error(y_test, y_pred))

import pickle
pickle.dump(pl_KNeighbors, open("stock_data\\pl_KNeighbors.p", "wb" ))


# ==============================================================================
# Cell 42
# ==============================================================================
# Again the results are fickle depending on the train/test split. Better to see statistical results.
#
# ### How does performance change with K? Take a look at Validation curve
# Learning curve->See errors as you change the number of training rows
#
# Validation curve->See errors as you change one of the model parameters


# ==============================================================================
# Cell 43
# ==============================================================================

knn_validation_list = []
numNeighbours = [4,8,16,32,64,100]
runNum = 40

for i in numNeighbours:
    print('Trying K='+str(i))
    for j in range(0, runNum):
    #Get a new train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=42+j)
        pl_KNeighbors = Pipeline([
            ('Power Transformer', PowerTransformer()),
            ('KNeighborsRegressor', KNeighborsRegressor(n_neighbors=i))
            ]).fit(X_train, y_train)
        
        y_pred = pl_KNeighbors.predict(X_test)
        
        resultThisRun = [i,
            mean_squared_error(y_train, pl_KNeighbors.predict(X_train)),
            mean_squared_error(y_test, y_pred)]
        
        knn_validation_list.append(resultThisRun)

knn_validation_df=pd.DataFrame(knn_validation_list, 
                               columns=['numNeighbours',
                                        'trainError',
                                        'testError'])


# ==============================================================================
# Cell 44
# ==============================================================================

knn_validation_df


# ==============================================================================
# Cell 45
# ==============================================================================

# Get our results in a format we can easily plot
knn_results_list = []
numNeighboursAttempted = knn_validation_df['numNeighbours'].unique()
results_df = pd.DataFrame(index=numNeighboursAttempted) #Create a DataFrame of results

for i in numNeighboursAttempted:
    
    blNeighbours = knn_validation_df['numNeighbours']==i#boolean mask
    
    trainErrorsMean = knn_validation_df[blNeighbours]['trainError'].mean()
    trainErrorsStd = knn_validation_df[blNeighbours]['trainError'].std()
    testErrorsMean = knn_validation_df[blNeighbours]['testError'].mean()
    testErrorsStd = knn_validation_df[blNeighbours]['testError'].std()
    knn_results_list.append([trainErrorsMean, trainErrorsStd,
                            testErrorsMean, testErrorsStd])
    

knn_results_df = pd.DataFrame(knn_results_list,
                              columns=['trainErrorsMean','trainErrorsStd',
                                       'testErrorsMean','testErrorsStd'],
                             index=numNeighboursAttempted)


# ==============================================================================
# Cell 46
# ==============================================================================

knn_results_df


# ==============================================================================
# Cell 47
# ==============================================================================

knn_results_df['trainErrorsMean'].plot(style='-x', figsize=(7,4.5))
knn_results_df['testErrorsMean'].plot(style='-x')

plt.fill_between(knn_results_df.index,\
                 knn_results_df['trainErrorsMean']
                 -knn_results_df['trainErrorsStd'],
                 knn_results_df['trainErrorsMean']
                 +knn_results_df['trainErrorsStd'], alpha=0.2)
plt.fill_between(results_df.index,\
                 knn_results_df['testErrorsMean']
                 -knn_results_df['testErrorsStd'],
                 knn_results_df['testErrorsMean']
                 +knn_results_df['testErrorsStd'], alpha=0.2)
plt.grid()
plt.legend(['Training CV MSE Score','Test Set MSE'])
plt.ylabel('MSE', fontsize=15)
plt.xlabel('Values for K', fontsize=15)
plt.title('K-NN Validation Curve', fontsize=20)
#plt.ylim([0, 0.8]);


# ==============================================================================
# Cell 48
# ==============================================================================
# ### Plot the Scatter Graph
# See if there is any predictive capability


# ==============================================================================
# Cell 49
# ==============================================================================

plt.figure(figsize=(6,6))
plotDensityContourPredVsReal('pl_KNeighbors', y_pred, y_test.to_numpy(),2)


# ==============================================================================
# Cell 50
# ==============================================================================
# Let's take a better look at the scatter graph data. Plotting a kernel density estimator contour over the top


# ==============================================================================
# Cell 51
# ==============================================================================
# ### Investigate predictive ability


# ==============================================================================
# Cell 52
# ==============================================================================

# See top 10 stocks and see how the values differ
PrintTopAndBottom10Predictions(y_test, y_pred)


# ==============================================================================
# Cell 53
# ==============================================================================

observePredictionAbility(pl_KNeighbors, X, y)


# ==============================================================================
# Cell 54
# ==============================================================================
# ### Investigate predictive ability with KDE plot


# ==============================================================================
# Cell 55
# ==============================================================================

# Output scatter plot and contour plot of density of points to see if prediciton matches reality
# Line of x=y is provided, perfect prediction would have all density on this line
# Also plot linear regression of the scatter

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde # Need for Kernel Density Calculation
from sklearn.linear_model import LinearRegression

def plotDensityContourPredVsRealContour(model_name, x_plot, y_plot, ps):
#x_plot, y_plot = y_pred, y_test.to_numpy()
    resolution = 40
    # Make a gaussian kde on a grid
    k = kde.gaussian_kde(np.stack([x_plot,y_plot]))
    xi, yi = np.mgrid[-ps:ps:resolution*4j, -ps:ps:resolution*4j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    # Plotting scatter and contour plot
    plt.pcolormesh(xi, 
                   yi, 
                   zi.reshape(xi.shape), 
                   shading='gouraud', 
                   cmap=plt.cm.Greens)
    plt.contour(xi, yi, zi.reshape(xi.shape) )
    plt.scatter(x_plot, y_plot, s=1)
    # Plotting linear regression
    LinMod = LinearRegression()
    LinMod.fit(y_plot.reshape(-1, 1), x_plot.reshape(-1, 1),)
    xx=[[-2],[2]]
    yy=LinMod.predict(xx)
    plt.plot(yy,xx)
    # Plot formatting
    plt.grid()
    plt.axhline(y=0, color='r', label='_nolegend_')
    plt.axvline(x=0, color='r', label='_nolegend_')
    plt.xlabel('Predicted Return')
    plt.ylabel('Actual Return')
    plt.plot([-100,100],[-100,100],'y--')
    plt.xlim([-ps*0.2,ps*0.6]) #
    plt.ylim([-ps,ps])
    plt.title('Predicted/Actual density plot for {}'.format(model_name))
    plt.legend([
        'Linear Fit Line','y=x Perfect Prediction Line',
        'Prediction Points'])
    # Save Figure
    #plt.savefig('result.png')


# ==============================================================================
# Cell 56
# ==============================================================================

plt.figure(figsize=(7,7))
plotDensityContourPredVsRealContour('pl_KNeighbors', y_pred, y_test.to_numpy(),1)


# ==============================================================================
# Cell 57
# ==============================================================================
# # Support Vector Machine Regression
# Warning: From grid search onward this takes a VERY long time and doesn't do that well relative to training time.
# ### Quick SVM Regressor with default parameters


# ==============================================================================
# Cell 58
# ==============================================================================

# Read in data and do train/test split.
X, y = loadXandyAgain()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print('X training matrix dimensions: ', X_train.shape)
print('X testing matrix dimensions: ', X_test.shape)
print('y training matrix dimensions: ', y_train.shape)
print('y testing matrix dimensions: ', y_test.shape)


# ==============================================================================
# Cell 59
# ==============================================================================

# SVM quick and dirty
from sklearn.svm import SVR

pl_svm = Pipeline([
    ('Power Transformer', PowerTransformer()),
    ('SVR', SVR()) # kernel='rbf', C=100, gamma=0.1, epsilon=.1 generated good returns. 
])

pl_svm.fit(X_train, y_train)
y_pred = pl_svm.predict(X_test)
from sklearn.metrics import mean_squared_error
print('mse: ', mean_squared_error(y_test, y_pred))
from sklearn.metrics import mean_absolute_error
print('mae: ', mean_absolute_error(y_test, y_pred))

import pickle
pickle.dump(pl_svm, open("stock_data\\pl_svm.p", "wb" ))


# ==============================================================================
# Cell 60
# ==============================================================================

observePredictionAbility(pl_svm, X, y)


# ==============================================================================
# Cell 61
# ==============================================================================
# ### As there are many parameters do a GridSearch CV, to find optimal parameters
# We aren't actually aiming for prediction ACCURACY though, we want stock return. Skip this part in the book.


# ==============================================================================
# Cell 62
# ==============================================================================

# Takes a LONG time
#Don't run this part unless you really want to grid search.
# best do in parallel

'''from sklearn.model_selection import GridSearchCV

parameters = [{'SVR__kernel': ['linear'],
               'SVR__C': [1, 10, 100],
               'SVR__epsilon': [0.05, 0.1, 0.2]},
                {'SVR__kernel': ['rbf'],
               'SVR__C': [1, 10, 100],
               'SVR__gamma': [0.001, 0.01, 0.1],
               'SVR__epsilon': [0.05, 0.1, 0.2]} ] 

# Best found was C=1, gamma=0.001, epsilon=0.2
svm_gs = GridSearchCV(pl_svm, 
                      parameters, 
                      cv=10, 
                      scoring='neg_mean_squared_error',
                      n_jobs = 4)#parallel

svm_gs.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(svm_gs.best_params_,'\n')
print("Grid scores on development set:")
means = svm_gs.cv_results_['mean_test_score']
stds = svm_gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svm_gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))'''


# ==============================================================================
# Cell 63
# ==============================================================================

# SVM with (supposedly) optimal parameters
from sklearn.svm import SVR
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

pl_svm = Pipeline([
    ('Power Transformer', PowerTransformer()),
    ('SVR', SVR(kernel='rbf', C=1, gamma=0.001, epsilon=.2)) # Now using optimal parameters (supposedly)
])

pl_svm.fit(X_train, y_train)
y_pred = pl_svm.predict(X_test)
from sklearn.metrics import mean_squared_error
print('mse: ', mean_squared_error(y_test, y_pred))
from sklearn.metrics import mean_absolute_error
print('mae: ', mean_absolute_error(y_test, y_pred))

import pickle
pickle.dump(pl_svm, open("stock_data\\pl_svm.p", "wb" ))

plt.figure(figsize=(6,6))
plotDensityContourPredVsReal('pl_svm', y_pred, y_test.to_numpy(), 2)


# ==============================================================================
# Cell 64
# ==============================================================================

observePredictionAbility(pl_svm, X, y)


# ==============================================================================
# Cell 65
# ==============================================================================
# ### Optimal prediction ability may not be optimal for US (Own selected parameters)
# yes the prediction coulld be more accurate, but if it is producing lower returns for us then the MSE measure of error isn't really what we want, even if it might tend towards what we want.


# ==============================================================================
# Cell 66
# ==============================================================================

def getMyPredictionAbility(my_pipeline, X, y):
    '''
    For a given predictor pipeline.
    Create table of top10 stock picks averaged, 
    and average that over several runs,
    to give us a synthetic performance result.    
    '''
    Top10PredRtrns, Top10ActRtrns=[], []
    Bottom10PredRtrns, Bottom10ActRtrns=[], []
    
    for i in range (0,10):
        # Pipeline and train/test
        X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.1, random_state=42+i)
        my_pipeline.fit(X_train, y_train)
        y_pred = my_pipeline.predict(X_test)
        
        # Put results in a DataFrame so we can sort it.
        y_results = pd.DataFrame()
        y_results['Actual Return'] = y_test
        y_results['Predicted Return'] = y_pred
        
        # Sort it by the prediced return.
        y_results.sort_values(by='Predicted Return',
                              ascending=False,
                              inplace=True)
        y_results.reset_index(drop=True,
                              inplace=True)

        # See top 10 stocks and see how the values differ
        Top10PredRtrns.append(round(
                              np.mean(
                              y_results['Predicted Return'].iloc[:10])
                              * 100, 2))
        Top10ActRtrns.append(round(
                             np.mean(
                             y_results['Actual Return'].iloc[:10])
                             * 100, 2))
        
        # See bottom 10 stocks and see how the values differ
        Bottom10PredRtrns.append(round(
                                 np.mean(
                                 y_results['Predicted Return'].iloc[-10:])
                                 * 100, 2))
        Bottom10ActRtrns.append(round(
                                np.mean(
                                y_results['Actual Return'].iloc[-10:])
                                * 100, 2))
        
        # View for debug
        #print([round( np.mean(y_results['Predicted Return'].iloc[:10])*100,2),
        #      round(np.mean(y_results['Actual Return'].iloc[:10])*100,2 ),
        #      round(np.mean(y_results['Predicted Return'].iloc[-10:])*100,2),
        #      round(np.mean(y_results['Actual Return'].iloc[-10:])*100,2 )])
    
    return round(np.mean(Top10ActRtrns),2), \
           round(np.array(Top10ActRtrns).std(),2)


# ==============================================================================
# Cell 67
# ==============================================================================

# Iterate through possible hyperparameters and find the combination that gives best return.
# Takes a long time
'''for kern in ['linear', 'rbf']:
    gam_val=0
    if kern == 'linear':
        for C_val in [1, 10, 100]:
            for eps_val in [0.05, 0.1, 0.2]:
                pl_svm = Pipeline([('Power Transformer', PowerTransformer()),\
                            ('SVR', SVR(kernel=kern, C=C_val, epsilon=eps_val)) ])
                performance, certainty = getMyPredictionAbility(pl_svm)
                print('return:', performance, 'std.dev.', certainty,\
                      'For Kernel:', kern, 'C:', C_val, 'Gamma:', gam_val, 'Epsilon', eps_val )
    if kern == 'rbf':
        for C_val in [1, 10, 100]:
            for eps_val in [0.05, 0.1, 0.2]:
                for gam_val in [0.001, 0.01, 0.1]:
                    pl_svm = Pipeline([('Power Transformer', PowerTransformer()),\
                                ('SVR', SVR(kernel=kern, C=C_val, gamma = gam_val, epsilon=eps_val)) ])      
                    performance, certainty = getMyPredictionAbility(pl_svm)
                    print('return:', performance, 'std.dev.', certainty,\
                          'For Kernel:', kern, 'C:', C_val, 'Gamma:', gam_val, 'Epsilon', eps_val )'''


# ==============================================================================
# Cell 68
# ==============================================================================

# SVM that gives higher stock returns.
from sklearn.svm import SVR

pl_svm = Pipeline([
    ('Power Transformer', PowerTransformer()),
    ('SVR', SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)) # best hyperparameter results
])

pl_svm.fit(X_train, y_train)
y_pred = pl_svm.predict(X_test)
from sklearn.metrics import mean_squared_error
print('mse: ', mean_squared_error(y_test, y_pred))
from sklearn.metrics import mean_absolute_error
print('mae: ', mean_absolute_error(y_test, y_pred))

import pickle
pickle.dump(pl_svm, open("stock_data\\pl_svm.p", "wb" ))


# ==============================================================================
# Cell 69
# ==============================================================================

plotDensityContourPredVsReal('pl_svm', y_pred, y_test.to_numpy(), 2)


# ==============================================================================
# Cell 70
# ==============================================================================

# See top 10 stocks and see how the values differ
PrintTopAndBottom10Predictions(y_test, y_pred)


# ==============================================================================
# Cell 71
# ==============================================================================

observePredictionAbility(pl_svm, X, y)


# ==============================================================================
# Cell 72
# ==============================================================================
# # Decision Tree Regression


# ==============================================================================
# Cell 73
# ==============================================================================

# Read in data and do train/test split.
X, y = loadXandyAgain()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print('X training matrix dimensions: ', X_train.shape)
print('X testing matrix dimensions: ', X_test.shape)
print('y training matrix dimensions: ', y_train.shape)
print('y testing matrix dimensions: ', y_test.shape)


# ==============================================================================
# Cell 74
# ==============================================================================

# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

pl_decTree = Pipeline([
    ('DecisionTreeRegressor',
     DecisionTreeRegressor(random_state=42)) # no need scaler
])

pl_decTree.fit(X_train, y_train)
y_pred = pl_decTree.predict(X_test)
from sklearn.metrics import mean_squared_error
print('train mse: ', 
      mean_squared_error(y_train, 
                         pl_decTree.predict(X_train)))
print('test mse: ',
      mean_squared_error(y_test, y_pred))

import pickle
pickle.dump(pl_decTree, open("stock_data\\pl_decTree.p", "wb" ))


# ==============================================================================
# Cell 75
# ==============================================================================
# That train error looks VERY low, it's likely overfitted. let's create a learning curve and find a good value for max_depth


# ==============================================================================
# Cell 76
# ==============================================================================

train_errors, test_errors, test_sizes=[], [], []

for i in range(2,50):
    pl_decTree = Pipeline([
    ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=42, max_depth=i))])
    pl_decTree.fit(X_train, y_train)
    y_pred = pl_decTree.predict(X_test)
    train_errors.append(mean_squared_error(y_train, pl_decTree.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, y_pred))
    test_sizes.append(i)

plt.plot(test_sizes, np.sqrt(train_errors),'r',test_sizes, np.sqrt(test_errors),'b')
plt.legend(['RMSE vs. training data','RMSE vs. testing data'])
plt.grid()
#plt.ylim([0,0.6])
plt.ylabel('RMSE');
plt.xlabel('max_depth');


# ==============================================================================
# Cell 77
# ==============================================================================
# We can't just choose a max_depth of 40+ because it is obviously overfitting there. The error vs. the training data is unfortunately not coming towards the error vs. testing data, so a lower tree depth is desirable, bt we can't choose a very low number, remember there are 15 columns to our x matrix (and possibly more if you have generated your own ratios).
#
# We'll settle on a tree_depth of 10. Any less and we will obviously be utilising too little of our data, yet the error between test and training sets is reasonably close.


# ==============================================================================
# Cell 78
# ==============================================================================
# ### Try max_depth of 10


# ==============================================================================
# Cell 79
# ==============================================================================

# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

pl_decTree = Pipeline([
    ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=42, max_depth=10))
])

pl_decTree.fit(X_train, y_train)
y_pred = pl_decTree.predict(X_test)
from sklearn.metrics import mean_squared_error
print('train mse: ', mean_squared_error(y_train, pl_decTree.predict(X_train)))
print('test mse: ', mean_squared_error(y_test, y_pred))

import pickle
pickle.dump(pl_decTree, open("stock_data\\pl_decTree.p", "wb" ))


# ==============================================================================
# Cell 80
# ==============================================================================

observePredictionAbility(pl_decTree, X, y)


# ==============================================================================
# Cell 81
# ==============================================================================

plt.figure(figsize=(5,5))
plotDensityContourPredVsReal('pl_decTree', y_pred, y_test.to_numpy(),2)


# ==============================================================================
# Cell 82
# ==============================================================================
# ### K-Fold cross validation
# We'll want to check how repeatable the decision tree regressor is.


# ==============================================================================
# Cell 83
# ==============================================================================

from sklearn.model_selection import cross_validate

pl_decTree = Pipeline([
    ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=42, max_depth=10))])

scores = cross_validate(pl_decTree, X, y, scoring='neg_mean_squared_error', cv=10, return_train_score=True)
print('K=10 Segments.')
print('Train scores', np.round(np.sqrt(-scores['train_score']), 2) )
print('Test scores', np.round(np.sqrt(-scores['test_score']), 2) )
print('-----------------')
print('AVERAGE TEST SCORE:', round(np.sqrt(-scores['test_score']).mean(),4),\
     'STD. DEV.:', round(np.sqrt(-scores['test_score']).std(),4))
print('AVERAGE TRAIN SCORE:', round(np.sqrt(-scores['train_score']).mean(),4),\
     'STD. DEV.:', round(np.sqrt(-scores['train_score']).std(),4))
print('-----------------')


# ==============================================================================
# Cell 84
# ==============================================================================
# Seems consistent, see if returns are consistent.


# ==============================================================================
# Cell 85
# ==============================================================================
# ### Investigate Predictive Ability
# What max tree depth shall we use?


# ==============================================================================
# Cell 86
# ==============================================================================

for my_depth in [4,5,6,7,8,9,10,15,20,30,50]:
    decTree = DecisionTreeRegressor(random_state=42, max_depth=my_depth) # no need scaler
    performance, certainty = getMyPredictionAbility(decTree, X, y)
    print('Tree max_depth:', my_depth, 
          'Average Return:',  performance,
          'Standard Deviation:', certainty)


# ==============================================================================
# Cell 87
# ==============================================================================
# ### Use max_depth of 15


# ==============================================================================
# Cell 88
# ==============================================================================

# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

pl_decTree = Pipeline([
    ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=42, max_depth=15))
])

pl_decTree.fit(X_train, y_train)
y_pred = pl_decTree.predict(X_test)
from sklearn.metrics import mean_squared_error
print('train mse: ', mean_squared_error(y_train, pl_decTree.predict(X_train)))
print('test mse: ', mean_squared_error(y_test, y_pred))

import pickle
pickle.dump(pl_decTree, open("stock_data\\pl_decTree.p", "wb" ))


# ==============================================================================
# Cell 89
# ==============================================================================

plt.figure(figsize=(5,5))
plotDensityContourPredVsReal('pl_decTree', y_pred, y_test.to_numpy(),2)


# ==============================================================================
# Cell 90
# ==============================================================================

PrintTopAndBottom10Predictions(y_test, y_pred)


# ==============================================================================
# Cell 91
# ==============================================================================

observePredictionAbility(pl_decTree, X, y)


# ==============================================================================
# Cell 92
# ==============================================================================

# See the decision tree
reg_decTree = DecisionTreeRegressor(random_state=42, max_depth=40)
reg_decTree.fit(X_train, y_train)
from sklearn import tree # Need this to see decision tree.
plt.figure(figsize=(10,10), dpi=400) # set figsize so we can see it
tree.plot_tree(reg_decTree, feature_names = X.keys(),  filled = True, max_depth=2, fontsize=10);
plt.savefig('RegDecTree.png')


# ==============================================================================
# Cell 93
# ==============================================================================
# # Random Forest Regression


# ==============================================================================
# Cell 94
# ==============================================================================

# Read in data and do train/test split.
X, y = loadXandyAgain()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print('X training matrix dimensions: ', X_train.shape)
print('X testing matrix dimensions: ', X_test.shape)
print('y training matrix dimensions: ', y_train.shape)
print('y testing matrix dimensions: ', y_test.shape)


# ==============================================================================
# Cell 95
# ==============================================================================

from sklearn.ensemble import RandomForestRegressor
#for my_depth in [4, 6, 10, 16, 24]: # faster
for my_depth in range(4,21):
    rForest = RandomForestRegressor(random_state=42, max_depth=my_depth) # no need scaler
    performance, certainty = getMyPredictionAbility(rForest, X, y)
    print('Tree max_depth:', my_depth, 
          'Average Return:', performance,
          'Standard Deviation:', certainty)


# ==============================================================================
# Cell 96
# ==============================================================================

from sklearn.ensemble import RandomForestRegressor
rfregressor = RandomForestRegressor(random_state=42, max_depth=10)
rfregressor.fit(X_train, y_train)
y_pred = rfregressor.predict(X_test)

print('train mse: ', 
      mean_squared_error(y_train, 
                         rfregressor.predict(X_train)))
print('test mse: ',
      mean_squared_error(y_test, 
                         y_pred))

import pickle
pickle.dump(rfregressor, open("stock_data\\pl_rfregressor.p", "wb" ))


# ==============================================================================
# Cell 97
# ==============================================================================

from sklearn.ensemble import ExtraTreesRegressor
ETregressor = ExtraTreesRegressor(random_state=42, max_depth=10)
ETregressor.fit(X_train, y_train)
y_pred_ET = ETregressor.predict(X_test)
print('train mse: ', 
      mean_squared_error(y_train, 
                         ETregressor.predict(X_train)))
print('test mse: ', 
      mean_squared_error(y_test, 
                         y_pred_ET))

import pickle
pickle.dump(ETregressor, open("stock_data\\pl_ETregressor.p", "wb" ))


# ==============================================================================
# Cell 98
# ==============================================================================

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plotDensityContourPredVsReal('rfregressor', y_pred, y_test.to_numpy(),2)
plt.title('Random Forest Regressor',fontsize=15)
plt.subplot(1,2,2)
plotDensityContourPredVsReal('ETregressor', y_pred_ET, y_test.to_numpy(),2)
plt.title('Extra (Random) Trees Regressor',fontsize=15)


# ==============================================================================
# Cell 99
# ==============================================================================

observePredictionAbility(rfregressor, X, y)


# ==============================================================================
# Cell 100
# ==============================================================================

observePredictionAbility(ETregressor, X, y)


# ==============================================================================
# Cell 101
# ==============================================================================

# Can see the importance of each feature in random forest.
ks, scores=[],[]
for k, score in zip(X.keys(), rfregressor.feature_importances_):
    print(k, round(score,3))
    ks.append(k)
    scores.append(score)


# ==============================================================================
# Cell 102
# ==============================================================================

plt.figure(figsize=(6,6))
plt.barh(ks,scores)
plt.grid()
plt.title('Random Forest Feature Relative Importance',fontsize=15)


# ==============================================================================
# Cell 103
# ==============================================================================
# ### Try a different way to visualise results


# ==============================================================================
# Cell 104
# ==============================================================================

from sklearn.ensemble import RandomForestRegressor
rfregressor = RandomForestRegressor(random_state=42, max_depth=10)
rfregressor.fit(X_train, y_train)
y_pred = rfregressor.predict(X_test)

y_results = pd.DataFrame()
y_results['Actual Return'] = y_test
y_results['Predicted Return'] = y_pred
y_results.sort_values(by='Predicted Return', ascending=False, inplace=True)
y_results.reset_index(drop=True, inplace=True)

endsNum = 100
plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
#plt.scatter(y_results.index[:endsNum].values, y_results['Predicted Return'].iloc[:endsNum], s=30)
plt.scatter(y_results.index[:endsNum].values, y_results['Actual Return'].iloc[:endsNum], s=30)
plt.grid()
plt.ylim([-1,5])
plt.title('Top '+str(endsNum)+' Predicted Stock Returns\n Plotted with Actual Returns', fontsize=20)
plt.ylabel('Stock Return', fontsize=15)
plt.xlabel('Sorted list line number\n (Highest Predictions to the left)', fontsize=15)
plt.subplot(1,2,2)
#plt.scatter(y_results.index[-endsNum:].values, y_results['Predicted Return'].iloc[-endsNum:], s=30)
plt.scatter(y_results.index[-endsNum:].values, y_results['Actual Return'].iloc[-endsNum:], s=30)
plt.grid()
plt.ylim([-1,5])
plt.title('Bottom '+str(endsNum)+' Predicted Stock Returns\n Plotted with Actual Returns', fontsize=20)
plt.ylabel('Stock Return', fontsize=15)
plt.xlabel('Sorted list line number\n (Lowest Predictions to the right)', fontsize=15)


# ==============================================================================
# Cell 105
# ==============================================================================
# # Gradient Boosted Decision Tree Regressor


# ==============================================================================
# Cell 106
# ==============================================================================

# Read in data and do train/test split.
X, y = loadXandyAgain()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print('X training matrix dimensions: ', X_train.shape)
print('X testing matrix dimensions: ', X_test.shape)
print('y training matrix dimensions: ', y_train.shape)
print('y testing matrix dimensions: ', y_test.shape)


# ==============================================================================
# Cell 107
# ==============================================================================

from sklearn.ensemble import GradientBoostingRegressor

pl_GradBregressor = Pipeline([
    ('GradBoostRegressor', GradientBoostingRegressor(n_estimators=100,\
                                                     learning_rate=0.1,\
                                                     max_depth=10,\
                                                     random_state=42,\
                                                     loss='squared_error'))\
])

pl_GradBregressor.fit(X_train, y_train)
y_pred = pl_GradBregressor.predict(X_test)
from sklearn.metrics import mean_squared_error
print('train mse: ', mean_squared_error(y_train, pl_GradBregressor.predict(X_train)))
print('test mse: ', mean_squared_error(y_test, y_pred))

import pickle
pickle.dump(pl_GradBregressor, open("stock_data\\pl_GradBregressor.p", "wb" ))


# ==============================================================================
# Cell 108
# ==============================================================================

plt.figure(figsize=(6,6))
plotDensityContourPredVsReal('pl_GradBregressor', y_pred, y_test.to_numpy(),2)


# ==============================================================================
# Cell 109
# ==============================================================================

# See top 10 stocks and see how the values differ for current train/test split
PrintTopAndBottom10Predictions(y_test, y_pred)


# ==============================================================================
# Cell 110
# ==============================================================================

observePredictionAbility(pl_GradBregressor, X, y)


# ==============================================================================
# Cell 111
# ==============================================================================
# # Plot all results


# ==============================================================================
# Cell 112
# ==============================================================================


# ==============================================================================
# Cell 113
# ==============================================================================

df_best, df_worst = pd.DataFrame(), pd.DataFrame()

# Run all our regressors (might take awhile.)
# We do this so we can plot easily with the pandas library.
# Make sure using the final hyperparameters.
df_best['LinearRegr'], df_worst['LinearRegr']=observePredictionAbility(
    pl_linear, X, y, returnSomething=True, verbose=False)

df_best['ElasticNet'], df_worst['ElasticNet']=observePredictionAbility(
    pl_ElasticNet, X, y, returnSomething=True, verbose=False)

df_best['KNN'], df_worst['KNN']=observePredictionAbility(
    pl_KNeighbors, X, y, returnSomething=True, verbose=False)

df_best['SVR'], df_worst['SVR']=observePredictionAbility(
    pl_svm, X, y, returnSomething=True, verbose=False)

df_best['DecTree'], df_worst['DecTree']=observePredictionAbility(
    pl_decTree, X, y, returnSomething=True, verbose=False)

df_best['RfRegr'], df_worst['RfRegr']=observePredictionAbility(
    rfregressor, X, y, returnSomething=True, verbose=False)

df_best['EtRegr'], df_worst['EtRegr']=observePredictionAbility(
    ETregressor, X, y, returnSomething=True, verbose=False)

df_best['GradbRegr'], df_worst['GradbRegr']=observePredictionAbility(
    pl_GradBregressor, X, y, returnSomething=True, verbose=False)


# ==============================================================================
# Cell 114
# ==============================================================================

# Plot results out
# Warning: the results are quite variable, 
# we are only looking at the means here.
# Comment out the code to see with standard deviations.

plt.figure(figsize=(14,8))
df_best.mean().plot(linewidth=0, 
                    marker='*', 
                    markersize=30, 
                    markerfacecolor='r', 
                    markeredgecolor='r', 
                    fontsize=16)
#(df_best.mean()+df_best.std()).plot(linewidth=0, marker='*', markersize=10, markerfacecolor='r', markeredgecolor='r')
#(df_best.mean()-df_best.std()).plot(linewidth=0, marker='*', markersize=10, markerfacecolor='r', markeredgecolor='r')
df_worst.mean().plot(linewidth=0, 
                     marker='o', 
                     markersize=25, 
                     markerfacecolor='b', 
                     markeredgecolor='b')
#(df_worst.mean()+df_worst.std()).plot(linewidth=0, marker='o', markersize=10, markerfacecolor='b', markeredgecolor='b')
#(df_worst.mean()-df_worst.std()).plot(linewidth=0, marker='o', markersize=10, markerfacecolor='b', markeredgecolor='b')

plt.legend(['Mean Top 10 Portfolios',
            'Mean Bottom 10 Portfolios'],
           prop={'size': 20})
#plt.ylim([-12, 60])
plt.title('Results of Selected Stock Portfolios Mean Performces,\n\
            Top10/Bottom 10 Stocks Each Portfolio,\n\
            Average of 10 Runs.', fontsize=20)
plt.ylabel('Return %', fontsize=20)
plt.grid()


# ==============================================================================
# Cell 115
# ==============================================================================
# # Try a hypothetical company and see the prediction


# ==============================================================================
# Cell 116
# ==============================================================================

X.mean()


# ==============================================================================
# Cell 117
# ==============================================================================

y.mean()


# ==============================================================================
# Cell 118
# ==============================================================================

# Put in a dataframe for prediction
avg_company = pd.DataFrame(X.mean().values.reshape(1,-1), columns=X.keys())


# ==============================================================================
# Cell 119
# ==============================================================================

rfregressor.predict(avg_company)


# ==============================================================================
# Cell 120
# ==============================================================================

# Try making a bunch of numbers reflect 
# higher earnings relative to price
good_company = pd.DataFrame(X.mean().values.reshape(1,-1), 
                            columns=X.keys())
good_company['EV/EBIT']=5
good_company['Op. In./(NWC+FA)']=0.4
good_company['P/E']=3
good_company['P/B']=4
good_company['P/S']=13
good_company['EBIT/TA']=1

rfregressor.predict(good_company)


# ==============================================================================
# Cell 121
# ==============================================================================

# Let's try the same in the opposite direction
bad_company = pd.DataFrame(X.mean().values.reshape(1,-1), columns=X.keys())
bad_company['EV/EBIT']=900
bad_company['Op. In./(NWC+FA)']=-0.5
bad_company['P/E']=30
bad_company['P/B']=300
bad_company['P/S']=400
bad_company['EBIT/TA']=0.04

rfregressor.predict(bad_company)
