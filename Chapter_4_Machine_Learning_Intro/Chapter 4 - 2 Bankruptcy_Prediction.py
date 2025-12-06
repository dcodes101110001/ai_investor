#!/usr/bin/env python3
# Converted from: Chapter 4 - 2 Bankruptcy_Prediction.ipynb


# ==============================================================================
# Cell 1
# ==============================================================================

# Code for the Book: International Stock Picking A.I. INVESTOR: A Guide to Build.
# Website: ai-investor.net
# Code needs am "All-In-One" subscription from the (presumably) nice people at https://eodhistoricaldata.com/. 
# Check the book text to see if they have changed their service since mid-2021.

# Damon Lee 2021

# Feel free to fork this code for others to see what can be done with it.


# ==============================================================================
# Cell 2
# ==============================================================================

import pandas as pd


# ==============================================================================
# Cell 3
# ==============================================================================
# # Bankruptcy Prediction Toy Example 
# ![image.png](attachment:image.png)
#
# ## Get The Data


# ==============================================================================
# Cell 4
# ==============================================================================

# Read in bad stocks and good stocks feature data
badStocks = pd.read_csv('badStocks.csv', index_col=0)
goodStocks = pd.read_csv('goodStocks.csv', index_col=0)

badStocks.head()


# ==============================================================================
# Cell 5
# ==============================================================================

goodStocks.head()


# ==============================================================================
# Cell 6
# ==============================================================================

# Isolate the specific features we want, and assign the outcomes
goodStocks['Bankrupt']=0
goodStocks = goodStocks[['Ticker',
                         'date',
                         '(CA-CL)/TA',
                         'RE/TA',
                         'EBIT/TA',
                         'BookEquity/TL', 
                         'Bankrupt']]

badStocks['Bankrupt']=1
badStocks = badStocks[['Ticker',
                       'date',
                       '(CA-CL)/TA',
                       'RE/TA',
                       'EBIT/TA',
                       'BookEquity/TL',
                       'Bankrupt']]
goodStocks.head()


# ==============================================================================
# Cell 7
# ==============================================================================

badStocks.head()


# ==============================================================================
# Cell 8
# ==============================================================================

# Combine the two DataFrames
stocksList = pd.concat([goodStocks, badStocks], ignore_index=True)
stocksList.drop(columns=['Ticker', 'date'], inplace=True)

# Extract the bankruptcy outcome labels to create the "y" outcomes
bankruptList = stocksList['Bankrupt']

# Keep only the stock feature data for the "X" feature data
stocksList.drop(columns=['Bankrupt'], inplace=True)

stocksList # View X


# ==============================================================================
# Cell 9
# ==============================================================================

bankruptList # view y


# ==============================================================================
# Cell 10
# ==============================================================================

stocksList.to_csv('bankruptStocks.csv')
bankruptList.to_csv('bankruptStocksTarget.csv')


# ==============================================================================
# Cell 11
# ==============================================================================
# ## Use Linear Discriminant Analysis - Like Altman 1968 Paper


# ==============================================================================
# Cell 12
# ==============================================================================

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis() # Create the model
clf.fit(stocksList, bankruptList) # Train(fit) the model

# Quick view of the results
print('Bankruptcy Prediction')
print(clf.predict(stocksList))
print('\n')
print('Actual Bankruptcy')
print(np.array(bankruptList))


# ==============================================================================
# Cell 13
# ==============================================================================

from matplotlib import pyplot as plt

# Visualising the bankruptcy prediction
plt.figure(figsize=(10,10), dpi=200)
plt.imshow([1-clf.predict(stocksList), 
            1-np.array(bankruptList)], cmap='gnuplot2')
plt.title('Bankruptcy Prediction With Linear Discriminant Analysis\n'+
          'Black means Predicting Bankrupt\n'+
          '(No Test/Train Split)')
plt.yticks(ticks=[0, 1], labels=['Prediction',
                                 'Actual'])
plt.xlabel('Company Stock (50 companies total)');


# ==============================================================================
# Cell 14
# ==============================================================================

print(clf.intercept_)
print(clf.coef_)


# ==============================================================================
# Cell 15
# ==============================================================================
# ![image.png](attachment:image.png)


# ==============================================================================
# Cell 16
# ==============================================================================
# ![image.png](attachment:image.png)


# ==============================================================================
# Cell 17
# ==============================================================================

badStocks.mean() # See the mean of the features


# ==============================================================================
# Cell 18
# ==============================================================================

goodStocks.mean() # See the mean of the features


# ==============================================================================
# Cell 19
# ==============================================================================
# ![image.png](attachment:image.png)


# ==============================================================================
# Cell 20
# ==============================================================================
# ## Testing On A Stock I Like (COST)


# ==============================================================================
# Cell 21
# ==============================================================================

# Taking SEC filings from http://investor.costco.com/
# Some rough figures from https://www.macrotrends.net/

# in $Millions
CA = 32565
CL = 31545
TA = 63078
RE = 5140 # Retained earnings
EBIT = 7438
TL = 43102
BookEquity = 19976

X1 = (CA-CL)/TA
X2 = RE/TA
X3 = EBIT/TA
X4 = BookEquity/TL


# ==============================================================================
# Cell 22
# ==============================================================================

# If predict bankrupt, returns 1, which is "True" in Python.
if clf.predict(np.array([X1,X2,X3,X4]).reshape(1,-1)):
    print('Going to go bust')
else:
    print('Not going bankrupt')


# ==============================================================================
# Cell 23
# ==============================================================================
# ## End


# ==============================================================================
# Cell 24
# ==============================================================================

# Not a useful plot, as we have 4 feature dimensions to consider, not 2. 
# Just looking at the data in a plot.

from matplotlib import pyplot as plt
plt.scatter(goodStocks['(CA-CL)/TA'], goodStocks['BookEquity/TL'])
plt.scatter(badStocks['(CA-CL)/TA'], badStocks['BookEquity/TL'], marker='X')
plt.xlabel('(CA-CL)/TA')
plt.ylabel('BookEquity/TL')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.grid()
plt.legend(['Non-Bankrupt Companies','Bankrupt Companies']);
plt.title('Bankrupt vs. Non-Bankrupt Companies\n With Two Features', fontsize=20);
