"""
Converted from: 1_Get_X_Y_Learning_Data_Raw.ipynb

This script contains code extracted from a Jupyter notebook.
Code has been organized for modular execution.
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from platform import python_version
from pandas.plotting import scatter_matrix
import yfinance as yf

"""
Combining Raw Data
"""
# Chapter 4 of the book: "Build Your Own AI Investor"

# Code from Book: Build Your Own AI Investor
# Damon Lee 2021
# Check out the performance on www.valueinvestingai.com
# Code uses data from the (presumably) nice people at https://simfin.com/. 
# Feel free to fork this code for others to see what can be done with it.

print(python_version())

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [7.0, 4.5]
plt.rcParams['figure.dpi'] = 150

#def getXDataMerged(myLocalPath='C:/Users/damon/Stock_Data/SimFin2023/'):
def getXDataMerged(myLocalPath='C:/Users/damon/OneDrive/BYO_Investing_AI/2024/Stock_Data/SimFin2024/'):
    '''
    For combining fundamentals financial data from SimFin,
    or SimFin+ (https://simfin.com/) without API.
    Download Income Statement, Balance Sheet and Cash Flow files,
    Place in a directory and give the directory path to the function.
    Assumes standard filenames from SimFin.
    Returns a DataFrame of the combined result. 
    Prints file infos.
    '''
    incomeStatementData = pd.read_csv(myLocalPath+'us-income-annual-full-asreported.csv', 
                delimiter=';')
    balanceSheetData = pd.read_csv(myLocalPath+'us-balance-annual-full-asreported.csv',
                delimiter=';')
    CashflowData = pd.read_csv(myLocalPath+'us-cashflow-annual-full-asreported.csv',
                delimiter=';')
    
    print('Income Statement CSV data is(rows, columns): ',
          incomeStatementData.shape)
    print('Balance Sheet CSV data is: ',
          balanceSheetData.shape)
    print('Cash Flow CSV data is: ' ,
          CashflowData.shape)
    

    # Merge the data together
    result = pd.merge(incomeStatementData, balanceSheetData,\
                on=['Ticker','SimFinId','Currency',
                    'Fiscal Year','Report Date','Publish Date'])
    
    result = pd.merge(result, CashflowData,\
                on=['Ticker','SimFinId','Currency',
                    'Fiscal Year','Report Date','Publish Date'])
    
    # dates in correct format
    result["Report Date"] = pd.to_datetime(result["Report Date"]) 
    result["Publish Date"] = pd.to_datetime(result["Publish Date"])
    
    print('Merged X data matrix shape is: ', result.shape)
    
    return result

def getYRawData(my_local_path='C:/Users/damon/OneDrive/BYO_Investing_AI/2024/Stock_Data/SimFin2024/'):
    '''
    Read stock price data from SimFin or SimFin+ (https://simfin.com/),
    without API.
    Place in a directory and give the directory path to the function.
    Assumes standard filenames from SimFin.
    Returns a DataFrame.
    Prints file info.
    '''
    dailySharePrices=pd.read_csv(my_local_path+
                                 'us-shareprices-daily.csv',
                                 delimiter=';')
    
    dailySharePrices["Date"]=pd.to_datetime(dailySharePrices["Date"])
    print('Stock Price data matrix is: ',dailySharePrices.shape)
    return dailySharePrices

def getYPriceDataNearDate(ticker, date, modifier, dailySharePrices):
    '''
    Return just the y price and volume.
    Take the first day price/volume of the list of days,
    that fall in the window of accepted days.
    'modifier' just modifies the date to look between.
    Returns a list.
    '''
    windowDays=5
    rows = dailySharePrices[
        (dailySharePrices["Date"].between(pd.to_datetime(date)
                                          + pd.Timedelta(days=modifier),
                                          pd.to_datetime(date)
                                          + pd.Timedelta(days=windowDays
                                                         +modifier)
                                         )
        ) & (dailySharePrices["Ticker"]==ticker)]
    
    if rows.empty:
        return [ticker, np.nan,\
                np.datetime64('NaT'),\
                np.nan]
    else:
        return [ticker, rows.iloc[0]["Open"],\
                rows.iloc[0]["Date"],\
                rows.iloc[0]["Volume"]*rows.iloc[0]["Open"]]

d=getYRawData()

getYPriceDataNearDate('AAPL', '2012-05-12', 0, d)

getYPriceDataNearDate('AAPL', '2023-02-06', 30, d)

def getYPricesReportDateAndTargetDate(x, d, modifier=365):
    '''
    Takes in all fundamental data X, all stock prices over time y,
    and modifier (days), and returns the stock price info for the
    data report date, as well as the stock price one year from that date
    (if modifier is left as modifier=365)
    '''
    # Preallocation list of list of 2 
    # [(price at date) (price at date + modifier)]
    y = [[None]*8 for i in range(len(x))] 
    
    whichDateCol='Publish Date'# or 'Report Date', 
    # is the performance date from->to. Want this to be publish date.
    
    # Because of time lag between report date
    # (which can't be actioned on) and publish date
    # (data we can trade with)
    
    # In the end decided this instead of iterating through index.
    # Iterate through a range rather than index, as X might not have
    # monotonic increasing index 1, 2, 3, etc.
    i=0
    for index in range(len(x)):
        y[i]=(getYPriceDataNearDate(x['Ticker'].iloc[index], 
                                    x[whichDateCol].iloc[index],0,d)
              +getYPriceDataNearDate(x['Ticker'].iloc[index], 
                                     x[whichDateCol].iloc[index], 
                                     modifier, d))
        i=i+1
        
    return y

'''def getYPricesReportDateAndTargetDate(x, d, modifier=365):
    '''
    #Takes in all fundamental data X, all stock prices over time y,
    #and modifier (days), and returns the stock price info for the
    #data report date, as well as the stock price one year from that date
    #(if modifier is left as modifier=365)
    '''
    # Preallocation list of list of 2 
    # [(price at date) (price at date + modifier)]
    y = [[None]*8 for i in range(len(x))] 
    
    whichDateCol='Publish Date'# or 'Report Date', 
    # is the performance date from->to. Want this to be publish date.
    
    # Because of time lag between report date
    # (which can't be actioned on) and publish date
    # (data we can trade with)

    for i in x.index:
        y[i]=(getYPriceDataNearDate(x['Ticker'].loc[i], 
                                    x[whichDateCol].loc[i],0,d)
              +getYPriceDataNearDate(x['Ticker'].loc[i], 
                                     x[whichDateCol].loc[i], 
                                     modifier, d))
        
    return y'''

X = getXDataMerged()
X.to_csv("Annual_Stock_Price_Fundamentals.csv")

for key in X.keys():
    print(key)

d = getYRawData()
d[d['Ticker']=='GOOG']

# \**Warning** takes a long time

# We want to know the performance for each stock, each year, between 10-K report dates.
# takes VERY long time, several hours,
y = getYPricesReportDateAndTargetDate(X, d, 365) # because of lookups in this function.

y = pd.DataFrame(y, columns=['Ticker', 'Open Price', 'Date', 'Volume',\
                             'Ticker2', 'Open Price2', 'Date2', 'Volume2'
                            ])
y.to_csv("Annual_Stock_Price_Performance.csv")

y

"""
Long part done, now remove rows with issues
"""

X=pd.read_csv("Annual_Stock_Price_Fundamentals.csv", index_col=0)
y=pd.read_csv("Annual_Stock_Price_Performance.csv", index_col=0)

X['Publish Date'] = pd.to_datetime(X['Publish Date'])
X[(X['Publish Date'] > '2004-01-01') & 
  (X['Publish Date'] < '2023-06-01')]['Publish Date'].hist(bins=200, 
                                                          figsize=(10,5))
plt.title('USA Financial Report Publication Dates');

# Find out things about data visually.
#X[X["Ticker"]=="GOOG"]['Income (Loss) from Continuing Operations'].hist()#bins=50, figsize=(20,15))
#X.keys()

#X.describe()
#X.hist(bins=50, figsize=(20,15))

for i in X.columns:
    print(i)

attributes=["Revenue","Net Income_x"]
scatter_matrix(X[attributes]);

# Find out things about Y data
print("y Shape:", y.shape)
print("X Shape:", X.shape)

y[(y['Volume']<1e4) | (y['Volume2']<1e4)]# rows with volume issues

# Now need to filter out rows because not all of the rows have stock performance.

## PROBLEMS
# no accounting for mergers or bankrupcies (use adjusted share closing price)

y.shape

# Issue where no share price
bool_list1 = ~y["Open Price"].isnull()
# Issue where there is low/no volume
bool_list2 = ~((y['Volume']<1e4) | (y['Volume2']<1e4))
# Issue where dates missing (Removes latest data too, which we can't use)
bool_list3 = ~y["Date2"].isnull()

y=y[bool_list1 & bool_list2 & bool_list3]
X=X[bool_list1 & bool_list2 & bool_list3]


# Issues where no listed number of shares
bool_list4 = ~X["Shares (Diluted)_x"].isnull()
y=y[bool_list4]
X=X[bool_list4]
               
y=y.reset_index(drop=True)
X=X.reset_index(drop=True)

y

X["Market Cap"] = y["Open Price"]*X["Shares (Diluted)_x"]

X.shape

y.shape

X.to_csv("Annual_Stock_Price_Fundamentals_Filtered.csv")
y.to_csv("Annual_Stock_Price_Performance_Filtered.csv")

y

X[80:90]

"""
X Data for Final Stock Selection 2024
"""
# Requires SimFin+ Bulk Download

# Get the Latest Share Prices (To make Market Cap. Column Soon)

def getYRawData2024(my_path = 'C:/Users/damon/OneDrive/BYO_Investing_AI/2024/Stock_Data/SimFin2024/'):
    d=pd.read_csv(my_path + 'us-shareprices-daily.csv', delimiter=';')
    d["Date"]=pd.to_datetime(d["Date"])
    print('Stock Price data matrix is: ',d.shape)
    return d

d=getYRawData2024()

getYPriceDataNearDate('AAPL', '2024-03-05', 0, d)

# Functions to Extract Price at Time We Need and Combine X Data

def getYPricesReportDate(X, d, modifier=365):
    '''
    Get the stock prices for our X matrix to create Market Cap. column later.
    '''
    i=0
    y = [[None]*8 for i in range(len(X))] # Preallocation list of list of 2 [(price at date) (price at date + modifier)]
    whichDateCol='Publish Date'# or 'Report Date', is the performance date from->to. Want this to be publish date.
    # Because of time lag between report date (which can't be actioned on) and publish date (data we can trade with)
    for index in range(len(X)):
        y[i]=getYPriceDataNearDate(X['Ticker'].iloc[index], X[whichDateCol].iloc[index], 0, d)
        i=i+1
    return y

def getXFullDataMerged(myLocalPath='C:/Users/damon/OneDrive/BYO_Investing_AI/2024/Stock_Data/SimFin2024/'):
    '''
    For combining fundamentals financial data from SimFin+ only,
    without API. 
    Download Income Statement, Balance Sheet and Cash Flow files,
    the -full versions, e.g. us-balance-annual-full.csv.
    Place in a directory and give the directory path to the function.
    Assumes standard filenames from SimFin.
    Returns a DataFrame of the combined result. 
    Prints file infos.
    '''
    incomeStatementData=pd.read_csv(myLocalPath+'us-income-annual-full-asreported.csv',
                                    delimiter=';')
    balanceSheetData=pd.read_csv(myLocalPath+'us-balance-annual-full-asreported.csv',
                                 delimiter=';')
    CashflowData=pd.read_csv(myLocalPath+'us-cashflow-annual-full-asreported.csv',
                             delimiter=';')
    
    print('Income Statement CSV data is(rows, columns): ',
          incomeStatementData.shape)
    print('Balance Sheet CSV data is: ',
          balanceSheetData.shape)
    print('Cash Flow CSV data is: ' ,
          CashflowData.shape)
    

    # Merge the data together
    result = pd.merge(incomeStatementData, balanceSheetData,\
                on=['Ticker','SimFinId','Currency',
                    'Fiscal Year','Report Date','Publish Date'])
    
    result = pd.merge(result, CashflowData,\
                on=['Ticker','SimFinId','Currency',
                    'Fiscal Year','Report Date','Publish Date'])
    
    # dates in correct format
    result["Report Date"] = pd.to_datetime(result["Report Date"]) 
    result["Publish Date"] = pd.to_datetime(result["Publish Date"])
    
    print('Merged X data matrix shape is: ', result.shape)
    
    return result

#OLD
'''#https://simfin.com/
def getXDataMerged():
    a=pd.read_csv('C:/Users/G50/Stock_Data/SimFin/2020/us-income-annual-full.csv', delimiter=';')
    b=pd.read_csv('C:/Users/G50/Stock_Data/SimFin/2020/us-balance-annual-full.csv', delimiter=';')
    c=pd.read_csv('C:/Users/G50/Stock_Data/SimFin/2020/us-cashflow-annual-full.csv', delimiter=';')
    print('Income Statement CSV is: ', a.shape)
    print('Balance Sheet CSV is: ', b.shape)
    print('Cash Flow CSV is: ' ,c.shape)
    result = pd.merge(a, b, on=['Ticker','SimFinId','Currency','Fiscal Year','Report Date','Publish Date'])
    result = pd.merge(result, c, on=['Ticker','SimFinId','Currency','Fiscal Year','Report Date','Publish Date'])
    result["Report Date"] = pd.to_datetime(result["Report Date"])
    result["Publish Date"] = pd.to_datetime(result["Publish Date"])
    print('merged X data matrix shape is:', result.shape)
    return result'''

# Get X Data (2023/24)

X = getXFullDataMerged()

X['Publish Date'].sort_values()

X['Publish Date'] = pd.to_datetime(X['Publish Date'])
X[(X['Publish Date'] > '2020-01-01') & 
  (X['Publish Date'] < '2023-03-07')]['Publish Date'].hist(bins=200, 
                                                          figsize=(10,5))
plt.title('USA Financial Report Publication Dates');

X['Publish Date'] = pd.to_datetime(X['Publish Date'])
X[(X['Publish Date'] > '2020-01-01') & 
  (X['Publish Date'] < '2023-03-18')]['Publish Date'].hist(bins=200, 
                                                          figsize=(10,5))
plt.title('USA Financial Report Publication Dates');

# Get data only for 2024
PublishDateStart = "2024-01-01"
PublishDateEnd = "2024-04-01"
bool_list = X['Publish Date'].between(\
              pd.to_datetime(PublishDateStart),\
              pd.to_datetime(PublishDateEnd) )
X=X[bool_list]

X

# Get Y Data

y = getYPricesReportDate(X, d) # Takes a min due to price lookups.
y = pd.DataFrame(y, columns=['Ticker', 'Open Price', 'Date', 'Volume'])

y

y['Ticker']

# Replace 'AAPL' with the ticker symbol of the stock you want to fetch data for
ticker_symbol = 'WMT'

# Create a yfinance object for the specified stock
stock = yf.Ticker(ticker_symbol)

# Get historical data for the stock with the 'history' method
# By default, it will return data for the last available trading day
historical_data = stock.history(period="1d")

# Extract the open price from the historical data
open_price = historical_data['Open'].iloc[0]

print(f"Open price for {ticker_symbol}: {open_price}")

openPrices = {}
for sym in y['Ticker']:
    # Replace 'AAPL' with the ticker symbol of the stock you want to fetch data for
    ticker_symbol = sym

    # Create a yfinance object for the specified stock
    stock = yf.Ticker(ticker_symbol)
    # Get historical data for the stock with the 'history' method
    # By default, it will return data for the last available trading day
    historical_data = stock.history(period="1d")
    if not historical_data.empty:
        # Extract the open price from the historical data
        open_price = historical_data['Open'].iloc[0]

        print(f"Open price for {ticker_symbol}: {open_price}")
        openPrices[ticker_symbol] = open_price

openPrices

# Filter Unwanted Rows with Both X and Y, Also Create Market Cap

y2 = pd.DataFrame(list(openPrices.items()), columns=['Ticker', 'Open Price'])
y2['Date']='2024-03-17'
y2

y2=y2.reset_index(drop=True)
X=X.reset_index(drop=True)

# Issue where no share price
bool_list1 = ~y["Open Price"].isnull()
# Issue where there is low/no volume
#bool_list2 = ~(y['Volume']<1e4)

y2=y2[bool_list1]
X=X[bool_list1]

# Issues where no listed number of shares
bool_list4 = ~X["Shares (Diluted)_x"].isnull()
y2=y2[bool_list4]
X=X[bool_list4]
               
y2=y2.reset_index(drop=True)
X=X.reset_index(drop=True)

X["Market Cap"] = y2["Open Price"]*X["Shares (Diluted)_x"]

X

y

# Save 2024 X Data for Stock Selection, Y Data for Ticker Names

#X.to_csv("Annual_Stock_Price_Fundamentals_Filtered_2023.csv")
#y.to_csv("Tickers_Dates_2023.csv")

X.to_csv("Annual_Stock_Price_Fundamentals_Filtered_2024_present.csv")
y2.to_csv("Tickers_Dates_2024_present.csv")
