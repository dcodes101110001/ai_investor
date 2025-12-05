"""
Converted from: Chapter 3 - Exercise 6 - Pandas Basics.ipynb

This script contains code extracted from a Jupyter notebook.
Code has been organized for modular execution.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt # Import plotting library
import pandas_datareader.data as pdr # Be sure this is installed from anaconda
import yfinance as yf

"""
Exercise 6 – Pandas Basics
"""

# Code from Book: Build Your Own AI Investor
# Damon Lee 2021
# Check out the performance on www.valueinvestingai.com
# Code uses data from the (presumably) nice people at https://simfin.com/. 
# Feel free to fork this code for others to see what can be done with it.

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [8.0, 5.0]
plt.rcParams['figure.dpi'] = 150

#Import Pandas

#Creating a dictionary with some table data in it.
dict = {
    'Name':['Tom', 'Dick', 'Harry'],
    'Age':[24, 35, 29],
    'Shoe Size':[7,8,9]
}

#Creating a DataFrame with the data from the dictionary
df=pd.DataFrame(dict)

#The dataframe can be viewed without the print() command.
df

#Write the data from the DataFrame to a file
df.to_csv('people.csv')

# Now I want to read the data, lets create a new DataFrame. Remember to specify an index column!
my_data=pd.read_csv('people.csv', index_col=0)

#have a look at the data
my_data

"""
Read in SimFin data
"""

# First Getting Stock Data
# Get csv data from [simfin.com](https://simfin.com/data/bulk). The following download settings were used.

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# Lets read some income statement data from SimFin (https://simfin.com/)
# Your download directory will differ.

# You can specify a full directory address
#fullDirectoryAddr = 'C:\Users\Damon\OneDrive\BYO_Investing_AI\2025\VIML_V7\JUPYTERNOTEBOOKS_collection_2024_working\Chapter_3_Intro_to_Python_Solutions\us-income-annual.csv'

# Or, if reading from current directory
currDirectoryAddr = 'us-income-annual.csv'

# Reading the .csv file into a DataFrame
Income_Data = pd.read_csv(currDirectoryAddr, 
                          delimiter=';')

# Get the DataFrame shape (Might be useful for other computation)
print('DataFrame shape is: ',Income_Data.shape)
# Have a look at what our DataFrame looks like
Income_Data
# You will notice that thefirst few and last few items are shown (Jupyter doesn't want to display all 15,124 rows.)

# The headings of columns are called keys. 
# The columns themselves are called series.
# To see what all the columns are in our dataframe, use the .keys() function. 
print(Income_Data.keys())

# To see series data for a single key:
Income_Data['Ticker']

Income_Data['Report Date']

print('The resulting type of object from our series selection is a:', 
      type(Income_Data['Report Date']) )

print('using .values after our selection turns this into a:', 
      type(Income_Data['Report Date'].values) )

# To get a list of tickers that exist, we could use the set operations we learned earlier.
my_set = set(Income_Data['Ticker'].values)
print(my_set)

len(my_set)

#Alternatively a dataframe has a function .unique()
Income_Data['Ticker'].unique()

# You can use logic on dataframes to obtain a list of boolean values that say whether a rule is True or False for each row.
# Let's try and isolate financial data for the 2017 fiscal period. 
# We will do a boolean check on items in the 'Fiscal Year' column:

IsIt2017 = (Income_Data['Fiscal Year'] == 2020)
print('The boolean logic on the Fiscal Year column is: ', IsIt2017)

# We can use this list of booleans to make a smaller dataframe with only data from fiscal year 2017 as follows:
Income_Data[IsIt2017]

# Or if we want to do this kind of filter in a single line:
Income_Data[Income_Data['Fiscal Year'] == 2020]
# Which is basically: 'I want the rows of Income_Data where the Income_Data Fiscal Year column is 2017.'
# Notice there are only 1,726 rows instead of 15,125

# Separate out all reports from 2017 only
# And company with profit above $1000.
IsIt2017 = (Income_Data['Fiscal Year'] == 2020)
HasIncome = (Income_Data['Net Income'] >= 1000)
Income_Data[HasIncome & IsIt2017]

# Let's order the stocks for financial year 2017 by net income.
# Within a DataFrame the rows can be ordered as follows:
Income_Data.sort_values(by=['Net Income'])

# However, we wish to perform this for only the 2017 fiscal year. 
# Fortunately you can just put one operations in front of the other as follows:
Income_Data[Income_Data['Fiscal Year'] == 2020].sort_values(by=['Net Income'])
# It looks like AAPL and BRKA earned the most money that year, more than $40bn!

# Lets plot the distribution of net income for all american companies:
Income_Data['Net Income'].hist()

# Plot graph with just 2020 FY data, logarithmic Y axis and more bins for greater resolution.

# Boolean mask for year 2020 data only.
Year2020Only = (Income_Data['Fiscal Year'] == 2020)

# Boolean mask for Companies with greater than -$10Bn profit
reasonableProfit = (Income_Data['Net Income'] >= -1e10)

# Use the booklean masks and plot the data
Income_Data[Year2020Only&reasonableProfit]['Net Income'].hist(bins=100, 
                                                              log=True)
plt.title('USA Corporate Net Income for 2020 Histogram')
plt.xlabel('Value in USD')
plt.ylabel('Number of Instances')

print('Max profit value was: $', Income_Data[Year2020Only&reasonableProfit]['Net Income'].max()/1e9,'Billion')

# Find the top 10 USA companies by net income, 2020.
Income_Data[Year2020Only&reasonableProfit][['Ticker',
                                            'Currency',
                                            'Fiscal Year', 
                                            'Report Date',
                                            'Publish Date',
                                            'Restated Date',
                                            'Net Income']].sort_values(by=['Net Income'], ascending=False).head(10)

# Statistical data can be calculated with the dataframe .describe() function
Income_Data.describe()

# You can create new dataframes from filtered old dataframes.
# Let's separate the 2017 fiscal year data to a new dataframe.
Income_Data_2020 = Income_Data[Income_Data['Fiscal Year'] == 2020]

Income_Data_2020.head(9)

Income_Data_2020.tail()

"""
Exercise 6
"""

# If stored the files in a different location
#incomeDataFile = 'C:/Users/G50/Stock_Data/SimFin/us-income-annual/us-income-annual.csv'
#stockPricesFile = 'C:/Users/G50/Stock_Data/SimFin/us-shareprices-daily/us-shareprices-daily.csv'

# If files in current directory
incomeDataFile = 'us-income-annual.csv'
stockPricesFile = 'us-shareprices-daily.csv'

Income_Data = pd.read_csv(incomeDataFile, 
                          delimiter=';')

stock_prices = pd.read_csv(stockPricesFile, 
                           delimiter=';')

# Preliminary View Of The Data

# How big are these dataframes? How is the data stored? 
print('Income data size is: ', Income_Data.shape)
print('stock_prices data size is: ', stock_prices.shape)

# How big are these dataframes? How is the data stored? 
print('Income data size is: ', Income_Data.shape)
print('stock_prices data size is: ', stock_prices.shape)

# What does the data look like? Use head() to take a look.
Income_Data.head(5) # See income statement data sample

stock_prices.head(5) # See stock price data sample

Income_Data.describe() # See some statistics about income statement data.

stock_prices.describe() # See some statistics about stock price data.

'''
We can understand a lot about the stock market with this data with some plots
Feel free to try different plots

Here I'm plotting the SG&A expense vs. the gross profit. 
We expect some kind of relationship between these.
'''

Income_Data.plot.scatter(x = 'Selling, General & Administrative', y = 'Gross Profit')
plt.grid()

# Exploring Some Companies From The Data

'''
Notice int he last plot that there is one company with a crazy high gross profit, 
yet having a very low SG&A spend, relatively speaking.

Any guesses for which company this is? And what year? Let's find out.
'''

Income_Data[['Ticker',
            'Currency',
            'Fiscal Year', 
            'Report Date',
            'Publish Date',
            'Restated Date',
            'Net Income',
            'Selling, General & Administrative',
            'Gross Profit']].sort_values(by='Gross Profit', ascending=False).head(10)

# Not really surprising that it's Amazon, and that the top 10 includes Wall Mart and Apple.

# Let's take a look at some data from AMZN by itelf.
Income_Data[Income_Data['Ticker']=='AMZN']['Revenue']

# We can plot this data and see how the revenue and income changed over time.
# Amazon seems to grow revenues steadily, but the income grows much faster proportionally.
# Documentation for bar charts is here: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.bar.html

Income_Data[Income_Data['Ticker']=='AMZN'].plot.bar(x='Report Date', 
                                                    y=['Revenue',
                                                       'Net Income'], 
                                                    rot=0, 
                                                    subplots=True,
                                                    grid='on',
                                                    title='Amazon Revenue and Net Income');

# Let's try a slow growth company, Wall Mart.
Income_Data[Income_Data['Ticker']=='WMT'].plot.bar(x='Report Date', 
                                                    y=['Revenue',
                                                       'Net Income'], 
                                                    rot=0, 
                                                    subplots=True,
                                                    grid='on',
                                                    title='Wall Mart Revenue and Net Income');

# Working Out A Market Capitalisation Figure Manually

'''
In fundamentals investing, you'll want an idea of the valuation of a company
relative to something else, e.g. the earnings, or the value of the assets.
This would be reflected in ratios like the Price/Earnings ratio, etc.
This requires a Market Cap figure, which we don't have currently.

We'll work out a market cap figure here manually, in the process trying out some pandas functions.

We'll do this for 2020 only, and to do it we will take the number of shares outstanding,
and multiply this by the share price at the time.

Bear in mind that the share prices and number of shares outstanding change over time,
so this market cap figure and any ratios derived from it will be rough numbers.

Another limitation is that there might not be a stock price reported for that specific day
and we aren't finding the closest days price to use at this point (we will do later).
''';

# Get the date format set correctly.
Income_Data['Publish Date']=pd.to_datetime(Income_Data['Publish Date'])

# Plot a histogram with a large number of bins to get an idea of when most reporting is done.
Income_Data[Income_Data['Fiscal Year'] == 2019]['Publish Date'].hist(bins=100);
plt.title('USA stocks Annual Report Publish Date\n For Fiscal Year 2019\n'+
          str(len(Income_Data[Income_Data['Fiscal Year'] == 2019]['Ticker'].unique()))+
          ' Stocks In Total');
plt.xlabel('Publish Date');
plt.ylabel('Number of Instances');

# Make a new dataframe of only the income data from 'Fiscal Year' == 2019.
Income_Data_2019 = Income_Data[Income_Data['Fiscal Year'] == 2019]

# View the series of company tickers that exist for that fiscal year.
(Income_Data[Income_Data['Fiscal Year'] == 2019]['Ticker']).unique

# It appears as though most reporting is done just after the beginning of March for these companies.
# We need stock prices that correspond to that date (roughly).
# Filter the stock price dataframe for the beginning of March.
stock_prices_2019 = (stock_prices[stock_prices['Date'] == '2020-03-02'])
print('stock_prices_2019 dataframe shape is: ', stock_prices_2019.shape)
stock_prices_2019.head(10)

# Now checking the data before we do Shares Outstanding*Market Cap

# Number of fundamental stock data rows we have;
print(Income_Data[Income_Data['Fiscal Year'] == 2019]['Ticker'].shape[0])

# Number of stock price data rows we have:
print(stock_prices_2019.shape[0])

# Notice the size the fundamentals data and the stock price data don't match
# In the real work we frequently deal with imperfect data. 
# We will now cut down the data to make both dataframes correspond
# so that we can add a market cap column to create ratios like P/E, P/S etc.

# Use the .isin() dataframe function to first cut down the share prices dataframe to only contain stocks that are in the 
# Income statement dataframe.
stock_prices_2019 = stock_prices_2019[stock_prices_2019['Ticker'].isin(Income_Data_2019['Ticker'])]

# Do the same in the opposite direction. 
Income_Data_2019 = Income_Data_2019[Income_Data_2019['Ticker'].isin(stock_prices_2019['Ticker'])]
Income_Data_2019

# See the DataFrame
stock_prices_2019

Income_Data_2019 # Both stock_prices_2019 and Income_Data_2019 should have the same height

# We have lost some rows in our dataframes in keeping only common stock tickers.
# Arrange both income and stock price dataframes alphabetically so the rows correspond.
stock_prices_2019 = stock_prices_2019.sort_values(by=['Ticker'])
Income_Data_2019 = Income_Data_2019.sort_values(by=['Ticker'])

# Make the stock_data DataFrame from the Income Data and add a Market Cap series.
stock_data_2019 = Income_Data_2019
stock_data_2019['Market Cap'] = Income_Data_2019['Shares (Diluted)'].values * stock_prices_2019['Open'].values

# Take a look at the distribution of Market Caps in the stock market with a histogram.
# use a large number of hisogram bins and a log scale, this will enable you to view the data better.
# You will see that there are many at the low end, and very few super large corporations in America

stock_data_2019['Market Cap'].hist(bins=100, log=True);
# Try to format this graph yourself with title, axis labels and so on.

# Finding Stocks With Market Cap, P/E Ratio Screens and More

# Let's see which companies are the largest by market cap.
stock_data_2019[['Ticker',
                'Currency',
                'Fiscal Year', 
                'Report Date',
                'Publish Date',
                'Restated Date',
                'Net Income',
                'Selling, General & Administrative',
                'Gross Profit',
                'Market Cap']].sort_values(by=['Market Cap'], 
                                             ascending=False).head(10)

'''
It's no surprise that AAPL and AMZN are present, 
earlier we found that they had the largest gross profits.

When picking stocks with fundamentals we would like the most bang/buck.

Let's create a P/E ratio, and find some stocks using that.
'''

# Creating a P/E ratio, here we have clipped the result to +=1000.
stock_data_2019['P/E'] = (stock_data_2019['Market Cap']/stock_data_2019['Net Income']).clip(-1000,1000)
stock_data_2019['P/E'].hist(bins=100, log=True);

# Let's see which companies are lowest by P/E ratio.
# We want t remove stocks with P/E < 0 too.

stock_data_2019[['Ticker',
                'Currency',
                'Fiscal Year', 
                'Report Date',
                'Publish Date',
                'Restated Date',
                'Net Income',
                'Selling, General & Administrative',
                'Gross Profit',
                'Market Cap',
                'P/E']][stock_data_2019['P/E'] > 0].sort_values(by=['P/E'],
                                    ascending=True).head(10)

# There are some companies with a market cap of less than $10 million, which is very low.
# A lot of people consider these micro-cap stocks to be very risky, 
# so it might be a good idea to filter them out


# RHE is at the top of the list. 
# With a low P/E ratio, the market seems quite pessimistic abouyt this company.
# Let's see if that's justified by plotting the revenue and earnings over time.
Income_Data[Income_Data['Ticker']=='RHE'].plot.bar(x='Report Date', 
                                                    y=['Revenue',
                                                       'Net Income'], 
                                                    rot=0, 
                                                    subplots=True,
                                                    grid='on',
                                                    title='RHE Revenue and Net Income');

# The revenue seems to be declining steadily, which doesn't look good.
# The P/E is an obvious measure of value, so it's not a surprise that stocks with low P/E
# look pretty bad.

# End of Exercise 6.

"""
Pandas Data Reader
"""

!pip install yfinance --upgrade --no-cache-dir

pip install git+https://github.com/ranaroussi/yfinance.git@hotfix/download-database-error

def get_data_yahoo(Ticker, start, end):
    stock = yf.Ticker(Ticker)
    data = stock.history(start = start, end= end)
    data.index = pd.to_datetime(data.index.date)
    # Function implementation here
    return data
pdr.get_data_yahoo = get_data_yahoo

# Start and end dates
start = pd.to_datetime('2019-05-01')
end = pd.to_datetime('2020-05-01')

# Use the Pandas DataReader
#tickerData = pdr.DataReader('SPY', 'yahoo', start, end); # Sometimes things stop working. Such is life.

tickerData = pdr.get_data_yahoo("SPY", start, end)

# Plot the data
plt.plot(tickerData['Open']);
plt.grid();
plt.xlabel('Date');
plt.ylabel('SPY value');

tickerData

# iloc and loc

# .iloc[] selects rows like a list

tickerData.iloc[-5:] #last 5 rows

tickerData.iloc[:5] #first 5 rows

tickerData.iloc[207] # row 207

# .loc[] selects rows using the index. 
# With Timeseries data like this, the index is a date, not a number.
tickerData.loc['2019-05-01']

tickerData.loc[tickerData.index>'2020-01-01']['Close'].plot()
plt.grid() # Add a grid to the plot.
