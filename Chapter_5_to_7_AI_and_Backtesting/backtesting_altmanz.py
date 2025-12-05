"""
Converted from: 4_Backtesting_AltmanZ.ipynb

This script contains code extracted from a Jupyter notebook.
Code has been organized for modular execution.
"""

import pandas as pd
import numpy as np
import math
import pickle # get the ML model from other notebook
from matplotlib import pyplot as plt # scatter plot
import matplotlib.lines as mlines # plot
from platform import python_version
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt

"""
Backtesting the AI Investor
"""
# Chapter 5 of the book: "Build Your Own AI Investor"

# Code from Book: Build Your Own AI Investor
# Damon Lee 2021
# Check out the performance on www.valueinvestingai.com
# Code uses data from the (presumably) nice people at https://simfin.com/. 
# Feel free to fork this code for others to see what can be done with it.

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [7.0, 4.5]
plt.rcParams['figure.dpi'] = 150

print(python_version())

def loadXandyAgain(randRows=False):
    '''
    Load X and y.
    Randomises rows.
    Returns X, y.
    '''
    # Read in data
    X=pd.read_csv("Annual_Stock_Price_Fundamentals_Ratios.csv",
                  index_col=0)
    y=pd.read_csv("Annual_Stock_Price_Performance_Percentage.csv",
                  index_col=0)
    y=y["Perf"] # We only need the % returns as target
    
    if randRows:
        # randomize the rows
        X['y'] = y
        X = X.sample(frac=1.0, random_state=42) # randomize the rows
        y = X['y']
        X.drop(columns=['y'], inplace=True)

    return X, y

"""
Train a model from here for backtest
"""
# Otherwise train a model in previous notebook, where data will be loaded into this notebook.
#
# We select stocks in a backtest with a picked, pretrained model.
#
# The the train set trains the mdoel the test set is sent to the backtester.

X, y = loadXandyAgain()

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.5, 
                                                    random_state=42)

# Save CSVs 
# For the backtester to get correct test data
# in case want to see the data.
X_train.to_csv("Annual_Stock_Price_Fundamentals_Ratios_train.csv")
X_test.to_csv("Annual_Stock_Price_Fundamentals_Ratios_test.csv")
y_train.to_csv("Annual_Stock_Price_Performance_Percentage_train.csv")
y_test.to_csv("Annual_Stock_Price_Performance_Percentage_test.csv")

# Linear

pl_linear = Pipeline([('Power Transformer', PowerTransformer()),
    ('linear', LinearRegression())]).fit(X_train, y_train)

y_pred = pl_linear.predict(X_test)

print('train mse: ', 
      mean_squared_error(y_train, pl_linear.predict(X_train)))
print('test mse: ',
      mean_squared_error(y_test, y_pred))

pickle.dump(pl_linear, open("pl_linear.p", "wb" ))

# Or can use Random Forest

X, y = loadXandyAgain()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Save CSVs 
# For the backtester to get correct test data
# in case want to see the data.
X_train.to_csv("Annual_Stock_Price_Fundamentals_Ratios_train.csv")
X_test.to_csv("Annual_Stock_Price_Fundamentals_Ratios_test.csv")
y_train.to_csv("Annual_Stock_Price_Performance_Percentage_train.csv")
y_test.to_csv("Annual_Stock_Price_Performance_Percentage_test.csv")

# Forest

rfregressor = RandomForestRegressor(random_state=42, max_depth=10).fit(X_train, y_train)

y_pred = rfregressor.predict(X_test)

print('train mse: ', mean_squared_error(y_train, rfregressor.predict(X_train)))
print('test mse: ', mean_squared_error(y_test, y_pred))
pickle.dump(rfregressor, open("rfregressor.p", "wb" ))

"""
Read in Train/Test
"""

# X AND Y
# The backtester needs dates from the old y vector 
# to plot the stock prices.

# Financial ratios 
X=pd.read_csv("Annual_Stock_Price_Fundamentals_Ratios.csv", 
              index_col=0)

# Annual stock performances, with date data.
y_withData=pd.read_csv("Annual_Stock_Price_Performance_Filtered.csv", 
                       index_col=0)

# Convert to date
y_withData["Date"] = pd.to_datetime(y_withData["Date"])
y_withData["Date2"] = pd.to_datetime(y_withData["Date2"])

# X AND Y (splitting for train/test done previously for trained model)
X_train=pd.read_csv("Annual_Stock_Price_Fundamentals_Ratios_train.csv", 
                    index_col=0)
X_test=pd.read_csv("Annual_Stock_Price_Fundamentals_Ratios_test.csv", 
                   index_col=0)
y_train=pd.read_csv("Annual_Stock_Price_Performance_Percentage_train.csv", 
                    index_col=0)
y_test=pd.read_csv("Annual_Stock_Price_Performance_Percentage_test.csv", 
                   index_col=0)

# Get y_withData to correspond to y_test
y_withData_Test=pd.DataFrame()
y_withData_Test=y_withData.loc[y_test.index, :]

# Convert string to datetime
y_withData_Test["Date"] = pd.to_datetime(y_withData_Test["Date"])
y_withData_Test["Date2"] = pd.to_datetime(y_withData_Test["Date2"])

y_test.head() # y targets

y_withData_Test.head() # y data corresponding to y targets

"""
Z score to account for default chance
"""

def calcZScores(X):
    '''
    Calculate Altman Z'' scores 1995
    '''
    Z = pd.DataFrame()
    Z['Z score'] = 3.25 \
    + 6.51 * X['(CA-CL)/TA']\
    + 3.26 * X['RE/TA']\
    + 6.72 * X['EBIT/TA']\
    + 1.05 * X['Book Equity/TL']
    return Z

z = calcZScores(X)
z.head()

"""
Backtest Program
"""

# Daily stock price time series for ALL stocks. 5M rows. Some days missing.
def getYRawData(directory='C:/Users/damon/OneDrive/BYO_Investing_AI/2024/Stock_Data/SimFin2024/'):
    '''
    Can set directory to look for file in.
    Get daily stock price time series for ALL stocks. 
    5M rows. Some days missing.
    Returns DataFrame
    '''
    daily_stock_prices=pd.read_csv(directory+'us-shareprices-daily.csv',
                                   delimiter=';')
    daily_stock_prices["Date"]=pd.to_datetime(daily_stock_prices["Date"])
    print('Reading historical time series stock data, matrix size is: ', 
          daily_stock_prices.shape)
    return daily_stock_prices

def getYPerf(y_):
    y=pd.DataFrame()
    y["Ticker"] = y_["Ticker"]
    y["Perf"]=(y_["Open Price2"]-y_["Open Price"])/y_["Open Price"]
    y[y["Perf"].isnull()]=0
    return y

def getStockPriceBetweenDates(date1, date2, ticker, d, rows):
#     # Alternative way
#     rows = d[(d["Date"].between(pd.to_datetime(date1),\
#                                 pd.to_datetime(date2) )) \
#                                  & (d["Ticker"]==ticker)]
    rows = d.loc[(d["Date"]>date1) &\
                 (d["Date"]<date2) &\
                 (d["Ticker"]==ticker)]
    return rows

# Example stock price lookup
getStockPriceBetweenDates(pd.to_datetime('2010-03-07'), 
                          pd.to_datetime('2011-02-27'), 
                          'NOTV', daily_stock_prices_data, pd.DataFrame())['Close'].plot()

def getStockPriceData(dateTimeIndex, ticker, y_withData, mask, daily_stock_prices, rows):
    '''
    Get the stock price for a ticker
    between the buy/sell date (using y_withdata)
    2021 version change to select from March to March only, 
    go for more corresponding backtest to reality,
    rather than attampting to match the training data closely.
    '''
    #date1 = y_withData[mask][y_withData[mask]["Ticker"] == ticker]["Date"].values[0]
    #date2 = y_withData[mask][y_withData[mask]["Ticker"] == ticker]["Date2"].values[0]
    date1 = dateTimeIndex[0]
    date2 = dateTimeIndex[-1]
    rows = getStockPriceBetweenDates(date1, date2,\
                                     ticker, daily_stock_prices, rows)
    return rows

def getDataForDateRange(date_Index_New, rows):
    '''
    Given a date range(index), and a series of rows,
    that may not correspond exactly,
    return a DataFrame that gets rows data,
    for each period in the date range(index)
    '''
    WeeklyStockDataRows = pd.DataFrame()
    for I in date_Index_New:
        WeeklyStockDataRows = pd.concat([WeeklyStockDataRows, 
        rows.iloc[rows.index.get_indexer([pd.to_datetime(I)], method="nearest")]], 
                                         ignore_index=True)
    return WeeklyStockDataRows

def getStockTimeSeries(dateTimeIndex, y_withData, 
                       tickers, mask, daily_stock_prices):
    '''
    Get the stock price as a time series DataFrame
    for a list of tickers.
    A mask is used to only consider stocks for a certain period.
    dateTimeIndex is typically a weekly index,
    so we know what days to fetch the price for.
    '''
    stockRet = pd.DataFrame(index=dateTimeIndex)
    dTI_new = dateTimeIndex.strftime('%Y-%m-%d') # Change Date Format
    rows = pd.DataFrame()
    for tick in tickers:
        # Here "rows" is stock price time series data 
        # for individual stock
        rows = getStockPriceData(dateTimeIndex,
                                 tick, 
                                 y_withData, 
                                 mask, 
                                 daily_stock_prices, 
                                 rows)
        rows.index = pd.DatetimeIndex(rows["Date"])
        WeeklyStockDataRows = getDataForDateRange(dTI_new,
                                                  rows)
        # Here can use Open, Close, Adj. Close, etc. price
        stockRet[tick] = WeeklyStockDataRows["Close"].values
    return stockRet

# Example stock price series lookup
weeklyDateTimeIndex = pd.date_range(start='2019-03-01', 
                                    periods=52, 
                                    freq='W')

stocksPrices = getStockTimeSeries(weeklyDateTimeIndex, 
                                  y_withData='notused', 
                                  tickers=['AAPL', 'MSFT','TSLA'], 
                                  mask='notused',
                                  daily_stock_prices=daily_stock_prices_data)

stocksPrices.plot()

def getPortfolioRelativeTimeSeries(stockRet):
    '''
    Takes DataFrame of stock returns, one column per stock
    Normalises all the numbers so the price at the start is 1.
    Adds a column for the portfolio value.
    '''    
    for key in stockRet.keys():
        stockRet[key]=stockRet[key]/stockRet[key][0]
    stockRet["Portfolio"] = stockRet.sum(axis=1)/(stockRet.keys().shape[0])
    return stockRet

getPortfolioRelativeTimeSeries(stocksPrices).plot();

### First tutorial function reader will write for backtest, will add altmanZ score filter later in chapter. ###
def getPortTimeSeriesForYear(date_starting, y_withData, X, 
                             daily_stock_prices, ml_model_pipeline):
    '''
    Function runs a backtest.
    Returns DataFrames of selected stocks/portfolio performance,
    for 1 year.
    y_withData is annual stock performances (all backtest years)
    date_starting e.g. '2010-01-01'
    daily_stock_prices is daily(mostly) stock price time series for 
    all stocks
    '''
    
    # get y dataframe as ticker and ticker performance only
    y = getYPerf(y_withData)
    
    # Get performance only for stock reports
    # published in the time frame we care about,
    # mask original data using the start date
    # "Date" is the publication date.
    thisYearMask = y_withData["Date"].between(\
            pd.to_datetime(date_starting) - pd.Timedelta(days=60),\
            pd.to_datetime(date_starting))
    
    # Get return prediction from model
    y_pred = ml_model_pipeline.predict(X[thisYearMask])
    
    # Make it a DataFrame to select the top picks
    y_pred = pd.DataFrame(y_pred)
    
    # Bool list of top stocks
    bl_bestStocks=(y_pred[0]>y_pred.nlargest(8,0).tail(1)[0].values[0]) 
    
    # DatetimeIndex
    dateTimeIndex = pd.date_range(start=date_starting, 
                                  periods=52, 
                                  freq='W')
    
    # 7 greatest performance stocks of y_pred 
    ticker_list = y[thisYearMask].reset_index(drop=True)\
                  [bl_bestStocks]["Ticker"].values
    
    # Issue with one of the tickers equaling 0, fix with lambda function
    #ticker_list = list(filter(lambda dateTimeIndex: dateTimeIndex != 0, ticker_list))

    # After we know our stock picks, we get the stock performance
    # Get DataFrame index of time stamp, series of stock prices, keys=tickers
    stockRet = getStockTimeSeries(dateTimeIndex, y_withData, 
                                  ticker_list, thisYearMask, 
                                  daily_stock_prices)
    
    # Get DataFrame of relative stock prices from 
    # 1st day(or close) and whole portfolio
    stockRetRel = getPortfolioRelativeTimeSeries(stockRet)
    return [stockRetRel, stockRetRel["Portfolio"], ticker_list]

### Proper function ###
def getPortTimeSeriesForYear(date_starting, y_withData, X, 
                             daily_stock_prices, ml_model_pipeline):
    '''
    Function runs a backtest.
    Returns DataFrames of selected stocks/portfolio performance,
    for 1 year.
    y_withData is annual stock performances (all backtest years)
    date_starting e.g. '2010-01-01'
    daily_stock_prices is daily(mostly) stock price time series for
    all stocks
    '''
    
    # get y dataframe with ticker performance only
    y = getYPerf(y_withData)
    
    # Get performance only for time frame we care about,
    # mask original data using the start date
    thisYearMask = y_withData["Date"].between(\
              pd.to_datetime(date_starting) - pd.Timedelta(days=120),######
              pd.to_datetime(date_starting))
    
    
    # Get return prediction from model
    y_pred = ml_model_pipeline.predict(X[thisYearMask])
    
    # Make it a DataFrame to select the top picks
    y_pred = pd.DataFrame(y_pred)
    
    ##### Change in code for Z score filtering ##### 
    # Separate out stocks with low Z scores
    z = calcZScores(X)
    
    # 3.75 is approx. B- rating
    bl_safeStocks=(z['Z score'][thisYearMask].reset_index(drop=True)>2) 
    y_pred_z = y_pred[bl_safeStocks]
    
    # Get bool list of top stocks
    bl_bestStocks=(
        y_pred_z[0]>y_pred_z.nlargest(8,0).tail(1)[0].values[0]) 
    
    dateTimeIndex = pd.date_range(\
                          start=date_starting, periods=52, freq='W')
    
    # 7 greatest performance stocks of y_pred 
    ticker_list = \
    y[thisYearMask].reset_index(drop=True)\
                      [bl_bestStocks&bl_safeStocks]["Ticker"].values
    ##### Change in code for Z score filtering ##### 
    
    # After we know our stock picks, we get the stock performance
    # Get DataFrame index of time stamp, series of stock prices, 
    # keys=tickers
    stockRet = getStockTimeSeries(dateTimeIndex, y_withData,
                                  ticker_list, thisYearMask, 
                                  daily_stock_prices)
    
    # Get DataFrame of relative stock prices from 1st day(or close) 
    # and whole portfolio
    stockRetRel = getPortfolioRelativeTimeSeries(stockRet)
    return [stockRetRel, stockRetRel["Portfolio"], ticker_list]

def getPortTimeSeries(y_withData, X, daily_stock_prices, ml_model_pipeline, verbose=True):
    '''
    Returns DataFrames of selected stocks/portfolio performance since 2009.
    Needs X and y(with data), the daily_stock_prices DataFrame,
    the model pipeline we want to test.
    X is standard X for model input.
    y_withData is the stock price before/after df with date information.
    Input X and y must be data that the model was not trained on.
    '''
    # set date range to make stock picks over
    dr=pd.date_range(start='2006-01-01', periods=16, freq='Y') + pd.to_timedelta('9w') # start every March
    # For each date in the date_range, make stock selections
    # and plot the return results of those stock selections
    port_perf_all_years = pd.DataFrame()
    perfRef=1 # performance starts at 1.
    for curr_date in dr:
        
        [comp, this_year_perf, ticker_list] = \
        getPortTimeSeriesForYear(curr_date, y_withData, X,\
                                 daily_stock_prices, ml_model_pipeline)
        
        if verbose: # If you want text output
            print("Backtest performance for year starting ",\
                  curr_date, " is:",\
                  round((this_year_perf.iloc[-1]-1)*100,2), "%")
            print("With stocks:", ticker_list)
            for tick in ticker_list:
                print(tick, "Performance was:",\
                      round((comp[tick].iloc[-1]-1)*100,2), "%" )
            print("---------------------------------------------")
        
        # Stitch performance for every year together
        this_year_perf = this_year_perf * perfRef
        #print(comp)
        port_perf_all_years = pd.concat([port_perf_all_years,\
                                         this_year_perf])
        
        perfRef = this_year_perf.iloc[-1]
    
    # Return portfolio performance for all years
    port_perf_all_years.columns = ["Indexed Performance"]
    return port_perf_all_years

"""
Run a Backtest
"""

daily_stock_prices_data=getYRawData()

y_withData_Test

X_test

trained_model_pipeline = pickle.load(open("rfregressor.p", "rb" ))

backTest = getPortTimeSeries(y_withData_Test, X_test, 
                         daily_stock_prices_data, 
                         trained_model_pipeline)
print('Performance is: ', 100 * (backTest["Indexed Performance"][-1]-1), '%')

plt.plot(backTest)
plt.grid()
plt.legend(['Portfolio Backtest Returns'])
plt.ylabel('Relative Performance');

open_price

# Replace 'AAPL' with the ticker symbol of the stock you want to fetch data for
ticker_symbol = '^GSPC'

# Create a yfinance object for the specified stock
stock = yf.Ticker(ticker_symbol)

# Start and end dates
start = pd.to_datetime(backTest.index[0])
end = pd.to_datetime(backTest.index[-1])

# Use the Pandas DataReader
spy = stock.history(start=start, end=end)
spy = spy.asfreq('W-MON', method='pad') # Weekly
spy['Relative'] = spy["Open"]/spy["Open"][0]


plt.figure(figsize=(8,5))
plt.plot(spy['Relative'])
plt.plot(backTest)
plt.grid()
plt.xlabel('Time')
plt.ylabel('Relative returns')
plt.legend(['S&P500 index performance', 'Linear Regressor Stock Picker'])
#plt.savefig('spy.png')
print('volatility of AI investor was: ', backTest['Indexed Performance'].diff().std()*np.sqrt(52))
print('volatility of S&P 500 was: ', spy["Relative"].diff().std()*np.sqrt(52))

plt.plot(backTest)
plt.grid()
plt.legend(['Portfolio Backtest Returns'])
plt.ylabel('Relative Performance');

"""
Investigating The Results
"""

x_=pd.read_csv("Annual_Stock_Price_Fundamentals_Filtered.csv",
               index_col=0)

date_starting = '2019-01-01'

thisYearMask = y_withData["Date"].between(\
        pd.to_datetime(date_starting) - pd.Timedelta(days=60),\
        pd.to_datetime(date_starting))

x_[thisYearMask][['Report Date', 'Publish Date']]

y_small=getYPerf(y_withData_Test)
# y_small is cut down version of y with stock returns only

# Create a boolean mask for the backtest year we are interested in
myDate = pd.to_datetime('2015-03-07 07:00:00')
mask2015 = y_withData_Test["Date"].between( pd.to_datetime(myDate)
                                       -pd.Timedelta(days=60), 
                                       pd.to_datetime(myDate))

#y_withData_Test[mask2015] # Checking the mask works
#X[mask2015]

# Load the model pipeline
ml_model_pipeline = pickle.load(open("pl_linear.p", "rb" ))
y_pred = ml_model_pipeline.predict(X_test[mask2015]) # Get stock performance predictions
y_pred = pd.DataFrame(y_pred) # Turn into DataFrame

plt.figure(figsize=(5,5))
# Now output graph.
plt.scatter(y_pred[0], y_small[mask2015]["Perf"], s=1)
# Formatting
plt.grid()
plt.axis('equal')
plt.title('Returns accuracy for {} backtest'.format(myDate.year))
plt.xlabel('Predicted Return')
plt.ylabel('Actual Return')
plt.axvline(c='blue', lw=1)
plt.axhline(c='blue', lw=1)
plt.savefig('result.png')
plt.axis([-1,1,-1,1]);

# Top stocks picked, and predicted performance.
bl = (y_pred[0] > y_pred.nlargest(8,0).tail(1)[0].values[0])

print("\nTop predicted perf. stocks picked are:")
print(y_small[mask2015].reset_index(drop=True)[bl]["Ticker"])
print("\nTop stocks predicted performance is:")
print(y_pred[bl])


print("\nActual performance was: ")
print(y_small[mask2015].reset_index(drop=True)[bl])

# Calc Altman Z score:
Z = 3.25 \
+ 6.51 * X_test[mask2015].reset_index(drop=True)[bl]['(CA-CL)/TA']\
+ 3.26 * X_test[mask2015].reset_index(drop=True)[bl]['RE/TA']\
+ 6.72 * X_test[mask2015].reset_index(drop=True)[bl]['EBIT/TA']\
+ 1.05 * X_test[mask2015].reset_index(drop=True)[bl]['Book Equity/TL']
print('\nZ scores:\n',Z)

# bool list of 7 greatest performance stocks of y_pred 
bl_bestStocks = (y_pred[0] > y_pred.nlargest(8,0).tail(1)[0].values[0])

# See what the performance is of the selection
print("Backtest return is:")
print(y_small[mask2015]["Perf"].reset_index(drop=True)[bl_bestStocks].values.mean())

#d = getYRawData()

y_withData_Test[mask2015]

dateTimeIndex2015 = pd.date_range(\
                    start='2015-03-07', 
                              periods=52, 
                              freq='W')
    
rows = getStockPriceData(dateTimeIndex2015, "BXC", y_withData_Test, mask2015, d, rows=pd.DataFrame())
plt.plot(rows["Date"], rows["Close"]) # Adj. Close
plt.grid(True)

rows

"""
See how individual stocks performed
"""
# With the program functions used individually

# Make X ticks standard, and grab stock prices as close to those points as possible for each stock (To track performance)

#DatetimeIndex
date_range = pd.date_range(start=myDate, periods=52, freq='W')

# 7 greatest performance stocks of y_pred 
ticker_list = y_withData_Test[mask2015].reset_index(drop=True)[bl_bestStocks]["Ticker"].values
stockRet = getStockTimeSeries(date_range, y_withData_Test, ticker_list , mask2015, d)

y_small

#make X ticks standard, and grab stock prices as close to
# those points as possible for each stock (To track performance)

#DatetimeIndex
date_range = pd.date_range(start=myDate, periods=52, freq='W') 

bl = (y_pred[0] > y_pred.nlargest(8,0).tail(1)[0].values[0])

# 7 greatest performance stocks of y_pred 
ticker_list = y_small[mask2015].reset_index(drop=True)[bl_bestStocks]["Ticker"].values

stockRet = getStockTimeSeries(date_range, y_withData_Test, 
                              ticker_list, 
                              mask2015, 
                              daily_stock_prices_data)

stockRetRel = getPortfolioRelativeTimeSeries(stockRet)

stockRetRel.head()

plt.plot(stockRetRel);
plt.grid()

stockRetRel = getPortfolioRelativeTimeSeries(stockRet)

stockRetRel.head(20)

plt.plot(stockRetRel);
plt.grid()

"""
Plot backtest with S&P500
"""

# GSPC.csv taken directly from Yahoo.com is the S&P500.
# https://finance.yahoo.com/quote/%5EGSPC/history?period1=1235174400&period2=1613865600&interval=1wk&filter=history&frequency=1wk&includeAdjustedClose=true
spy=pd.read_csv("GSPC.csv", index_col='Date', parse_dates=True)
spy = spy[spy.index > pd.to_datetime('2010-01-01')]
spy['Relative'] = spy["Open"]/spy["Open"][0]


plt.figure(figsize=(8,5))
plt.plot(spy['Relative'])
plt.plot(backTest)
plt.grid()
plt.xlabel('Time')
plt.ylabel('Relative returns')
plt.legend(['S&P500 index performance', 'Linear Regressor Stock Picker'])
#plt.savefig('spy.png')
print('volatility of AI investor was: ', backTest['Indexed Performance'].diff().std()*np.sqrt(52))
print('volatility of S&P 500 was: ', spy["Relative"].diff().std()*np.sqrt(52))

spy.iloc[-1]
