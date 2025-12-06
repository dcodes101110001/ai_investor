#!/usr/bin/env python3
# Converted from: 2_Process_X_Y_Learning_Data.ipynb


# ==============================================================================
# Cell 1
# ==============================================================================
# # Feature Engineering
# Chapter 4 of the book: "Build Your Own AI Investor"


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
# Cell 5
# ==============================================================================

# Code from Book: Build Your Own AI Investor
# Damon Lee 2021
# Check out the performance on www.valueinvestingai.com
# Code uses data from the (presumably) nice people at https://simfin.com/. 
# Feel free to fork this code for others to see what can be done with it.

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math


# ==============================================================================
# Cell 6
# ==============================================================================

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [7.0, 4.5]
plt.rcParams['figure.dpi'] = 150


# ==============================================================================
# Cell 7
# ==============================================================================
# # Read in Stock Data from last notebook


# ==============================================================================
# Cell 8
# ==============================================================================

# In this notebook we will use x_ for x fundamentals,
# and X is the input matrix we want at the end.
x_=pd.read_csv("Annual_Stock_Price_Fundamentals_Filtered.csv",
               index_col=0)
y_=pd.read_csv("Annual_Stock_Price_Performance_Filtered.csv",
               index_col=0)


# ==============================================================================
# Cell 9
# ==============================================================================
# # Get Fundamental Ratios


# ==============================================================================
# Cell 10
# ==============================================================================

def fixNansInX(x):
    '''
    Takes in x DataFrame, edits it so that important keys
    are 0 instead of NaN.
    '''
    keyCheckNullList = ["Short Term Debt" ,\
            "Long Term Debt" ,\
            "Interest Expense, Net",\
            "Income Tax (Expense) Benefit, Net",\
            "Cash, Cash Equivalents & Short Term Investments",\
            "Property, Plant & Equipment, Net",\
            "Revenue",\
            "Gross Profit",\
            "Total Current Liabilities"]
    x[keyCheckNullList]=x[keyCheckNullList].fillna(0)
    x['Property, Plant & Equipment, Net'] = x['Property, Plant & Equipment, Net'].fillna(0)
    
def addColsToX(x):
    '''
    Takes in x DataFrame, edits it to include:
        Enterprise Value.
        Earnings before interest and tax.
    
    '''
    x["EV"] = x["Market Cap"] \
    + x["Long Term Debt"] \
    + x["Short Term Debt"] \
    - x["Cash, Cash Equivalents & Short Term Investments"]

    x["EBIT"] = x["Net Income (Common)"] \
    - x["Interest Expense, Net"] \
    - x["Income Tax (Expense) Benefit, Net"]


# ==============================================================================
# Cell 11
# ==============================================================================

# Make new X with ratios to learn from.
def getXRatios(x_):
    '''
    Takes in x_, which is the fundamental stock DataFrame raw. 
    Outputs X, which is the data encoded into stock ratios.
    '''
    X=pd.DataFrame()
    
    # EV/EBIT
    X["EV/EBIT"] = x_["EV"] / x_["EBIT"]
    
    # Op. In./(NWC+FA)
    X["Op. In./(NWC+FA)"] = x_["Operating Income (Loss)"] \
    / (x_["Total Current Assets"] - x_["Total Current Liabilities"] \
       + x_["Property, Plant & Equipment, Net"])
    
    # P/E
    X["P/E"] = x_["Market Cap"] / x_["Net Income (Common)"]
    
    # P/B
    X["P/B"] = x_["Market Cap"] / x_["Total Equity"] 
    
    # P/S
    X["P/S"] = x_["Market Cap"] / x_["Revenue"] 
    
    # Op. In./Interest Expense
    X["Op. In./Interest Expense"] = x_["Operating Income (Loss)"]\
    / - x_["Interest Expense, Net"]
    
    # Working Capital Ratio
    X["Working Capital Ratio"] = x_["Total Current Assets"]\
    / x_["Total Current Liabilities"]
    
    # Return on Equity
    X["RoE"] = x_["Net Income (Common)"] / x_["Total Equity"]
    
    # Return on Capital Employed
    X["ROCE"] = x_["EBIT"]\
    / (x_["Total Assets"] - x_["Total Current Liabilities"] )
    
    # Debt/Equity
    X["Debt/Equity"] = x_["Total Liabilities"] / x_["Total Equity"]
    
    # Debt Ratio
    X["Debt Ratio"] = x_["Total Assets"] / x_["Total Liabilities"]
    
    # Cash Ratio
    X["Cash Ratio"] = x_["Cash, Cash Equivalents & Short Term Investments"]\
    / x_["Total Current Liabilities"]
    
    # Asset Turnover
    X["Asset Turnover"] = x_["Revenue"] / \
                            x_["Property, Plant & Equipment, Net"]
    
    # Gross Profit Margin
    X["Gross Profit Margin"] = x_["Gross Profit"] / x_["Revenue"]
    
    ### Altman ratios ###
    # (CA-CL)/TA
    X["(CA-CL)/TA"] = (x_["Total Current Assets"]\
                       - x_["Total Current Liabilities"])\
                        /x_["Total Assets"]
    
    # RE/TA
    X["RE/TA"] = x_["Retained Earnings"]/x_["Total Assets"]
    
    # EBIT/TA
    X["EBIT/TA"] = x_["EBIT"]/x_["Total Assets"]
    
    # Book Equity/TL
    X["Book Equity/TL"] = x_["Total Equity"]/x_["Total Liabilities"]
    
    X.fillna(0, inplace=True)
    return X

def fixXRatios(X):
    '''
    Takes in X, edits it to have the distributions clipped.
    The distribution clippings are done manually by eye,
    with human judgement based on the information.
    '''
    X["RoE"] = X["RoE"].clip(-5, 5)
    X["Op. In./(NWC+FA)"] = X["Op. In./(NWC+FA)"].clip(-5, 5)
    X["EV/EBIT"] = X["EV/EBIT"].clip(-500, 500)
    X["P/E"] = X["P/E"].clip(-1000, 1000)
    X["P/B"] = X["P/B"].clip(-50, 100)    
    X["P/S"] = X["P/S"].clip(0, 500)
    X["Op. In./Interest Expense"] = X["Op. In./Interest Expense"].clip(-600, 600)#-600, 600
    X["Working Capital Ratio"] = X["Working Capital Ratio"].clip(0, 30)  
    X["ROCE"] = X["ROCE"].clip(-2, 2)
    X["Debt/Equity"] = X["Debt/Equity"].clip(0, 100)
    X["Debt Ratio"] = X["Debt Ratio"].clip(0, 50)  
    X["Cash Ratio"] = X["Cash Ratio"].clip(0, 30)
    X["Gross Profit Margin"] = X["Gross Profit Margin"].clip(0, 1) #how can be >100%?
    X["(CA-CL)/TA"] = X["(CA-CL)/TA"].clip(-1.5, 2)
    X["RE/TA"] = X["RE/TA"].clip(-20, 2)
    X["EBIT/TA"] = X["EBIT/TA"].clip(-2, 1)
    X["Book Equity/TL"] = X["Book Equity/TL"].clip(-2, 20)
    X["Asset Turnover"] = X["Asset Turnover"].clip(-2000, 2000)# 0, 500


# ==============================================================================
# Cell 12
# ==============================================================================

def getYPerf(y_):
    '''
    Takes in y_, which has the stock prices and their respective
    dates they were that price.
    Returns a DataFrame y containing the ticker and the 
    relative change in price only.
    '''
    y=pd.DataFrame()
    y["Ticker"] = y_["Ticker"]
    y["Perf"]=(y_["Open Price2"]-y_["Open Price"])/y_["Open Price"]
    y["Perf"].fillna(0, inplace=True)
    return y


# ==============================================================================
# Cell 13
# ==============================================================================

from scipy.stats import zscore
def ZscoreSlice(ZscoreSliceVal):
    '''
    Slices the distribution acording to Z score.
    Any values with Z score above/below the argument will be given the max/min Z score value
    '''
    xz=x.apply(zscore) # Dataframe of Z scores   
    for key in x.keys():
        xps=ZscoreSliceVal * x[key].std()+x[key].mean()
        xns=ZscoreSliceVal * -x[key].std()+x[key].mean()
        x[key][xz[key]>ZscoreSliceVal]=xps
        x[key][xz[key]<-ZscoreSliceVal]=xns
    return x


# ==============================================================================
# Cell 14
# ==============================================================================
# ### Run the functions


# ==============================================================================
# Cell 15
# ==============================================================================

# From x_ (raw fundamental data) get X (stock fundamental ratios)
fixNansInX(x_)
addColsToX(x_)
X=getXRatios(x_)
fixXRatios(X)

# From y_(stock prices/dates) get y (stock price change)
y=getYPerf(y_)


# ==============================================================================
# Cell 16
# ==============================================================================

X


# ==============================================================================
# Cell 17
# ==============================================================================
# ### Before


# ==============================================================================
# Cell 18
# ==============================================================================

x_.head() # see x_


# ==============================================================================
# Cell 19
# ==============================================================================

y_.head() # see y_


# ==============================================================================
# Cell 20
# ==============================================================================
# ### After


# ==============================================================================
# Cell 21
# ==============================================================================

X.head() # see X


# ==============================================================================
# Cell 22
# ==============================================================================

y.head() # see y


# ==============================================================================
# Cell 23
# ==============================================================================
# ### See Distributions


# ==============================================================================
# Cell 24
# ==============================================================================

# See one of the distributions
k=X.keys()[2] # Try different numbers, 0-14.
X[k].hist(bins=100, figsize=(6,5))
plt.title(k);


# ==============================================================================
# Cell 25
# ==============================================================================

X.describe()


# ==============================================================================
# Cell 26
# ==============================================================================

# Make a plot of the distributions.
cols, rows = 3, 5
plt.figure(figsize=(5*cols, 5*rows))

for i in range(1, cols*rows):
    if i<len(X.keys()):
        plt.subplot(rows, cols, i)
        k=X.keys()[i]
        X[k].hist(bins=100)
        plt.title(k);


# ==============================================================================
# Cell 27
# ==============================================================================

y.to_csv("Annual_Stock_Price_Performance_Percentage.csv")
X.to_csv("Annual_Stock_Price_Fundamentals_Ratios.csv")


# ==============================================================================
# Cell 28
# ==============================================================================
# ### Try out power transformer see if our data has good distributions
# A lot of the algorithms won't work without appropriate transformation. We'll use the power transformer


# ==============================================================================
# Cell 29
# ==============================================================================

# Write code to plot out all distributions of X in a nice diagram
from sklearn.preprocessing import PowerTransformer
transformer = PowerTransformer()
X_t=pd.DataFrame(transformer.fit_transform(X), columns=X.keys())

def plotFunc(n, myDatFrame):
    myKey = myDatFrame.keys()[n]
    plt.hist(myDatFrame[myKey], density=True, bins=30)
    plt.grid()
    plt.xlabel(myKey)
    plt.ylabel('Probability')

plt.figure(figsize=(13,20))
plotsIwant=[4,6,9,10,11]

j=1
for i in plotsIwant:
    plt.subplot(len(plotsIwant),2,2*j-1)
    plotFunc(i,X)
    if j==1:
        plt.title('Before Transformation',fontsize=17)
    plt.subplot(len(plotsIwant),2,2*j)
    plotFunc(i,X_t)
    if j==1:
        plt.title('After Transformation',fontsize=17)
    j+=1
    
plt.savefig('Transformat_Dists.png', dpi=300)


# ==============================================================================
# Cell 30
# ==============================================================================
# # X Data for Final Stock Selection 2024/2023
# Requires SimFin PROBulk Download


# ==============================================================================
# Cell 31
# ==============================================================================

X=pd.read_csv("Annual_Stock_Price_Fundamentals_Filtered_2024_present.csv", 
              index_col=0)

# Net Income fix, checked annual reports.
X['Net Income (Common)'] = X['Net Income_x'] 

fixNansInX(X)
addColsToX(X)
X=getXRatios(X)
fixXRatios(X)
X.to_csv("Annual_Stock_Price_Fundamentals_Ratios_2024.csv")


# ==============================================================================
# Cell 32
# ==============================================================================

X
