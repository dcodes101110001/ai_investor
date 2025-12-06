#!/usr/bin/env python3
# Converted from: Chapter 3 - Exercise 3 - Logic and Loops.ipynb


# ==============================================================================
# Cell 1
# ==============================================================================
# # Exercise 3 - Logic and Loops


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
# # Exercise 3


# ==============================================================================
# Cell 4
# ==============================================================================

# Output stocks with a price/earnings ratio below a number and a beta above a number. 
# return a list of aceptable stocks with ascending by P/E
# Try and separate out these steps and work on one at a time, life is easier that way

# Toby has come and messed up the code! Try and get it working again by fixing the code missing where "?" appears


# ==============================================================================
# Cell 5
# ==============================================================================

# Given data.
Stocks = ['JKHY', 'FSTR', 'ETV', 'WAFD', 'RSG', 'HTGC']
PriceEarningsRatios = [48.54, 3.29, 5.71, 9.18, 23.89, 14.02]
Beta = [0.59, 1.67, 0.89, 0.99, 0.62, 1.51]


# ==============================================================================
# Cell 6
# ==============================================================================

'''
# Fix this function
def filterStocks(maxBeta, maxPE):
    filteredStocks = [?] # Create an empty list to be appended to.
    for i, stock in enumerate(Stocks): # This counts i as 1, 2, 3, ... down the list of stocks
        PriceEarningsBool = (PriceEarningsRatios[i] < ?) # returns True if our current PE is less than maxPE
        ? = (Beta[i] < ?) # returns True if our current beta is less than maxbeta
        if (? & BetaBool):
            filteredStocks.?(Stocks[i]) # with each loop, if conditions are met, append to the list.
    return ? # Return the filtered Stocks
'''

# Fix this function
def filterStocks(maxBeta, maxPE):
    '''
    This function reads global list variables:
    Stocks
    PriceEarningsRatios
    Beta
    
    and returns a list of stocks that are below the maxBeta and maxPE values
    '''
    filteredStocks = [] # Create an empty list to be appended to.
    for i, stock in enumerate(Stocks): # This counts i as 1, 2, 3, ... down the list of stocks
        PriceEarningsBool = (PriceEarningsRatios[i] < maxPE) # returns True if our current PE is less than maxPE
        BetaBool = (Beta[i] < maxBeta) # returns True if our current beta is less than maxbeta
        if (PriceEarningsBool & BetaBool):
            filteredStocks.append(Stocks[i]) # with each loop, if conditions are met, append to the list.
    return filteredStocks # Return the filtered Stocks


# ==============================================================================
# Cell 7
# ==============================================================================

# Try using this function 
maxBeta = 1.2 # Can try a bunch of numbers
maxPE = 20 # Can try a bunch of numbers

filteredStocks = filterStocks(maxBeta, maxPE)
print('Accepted filtered stocks are:', filteredStocks)


# ==============================================================================
# Cell 8
# ==============================================================================

# Try using this function 
maxBeta = 1.9
maxPE = 20

filteredStocks = filterStocks(maxBeta, maxPE)
print('Accepted filtered stocks are:', filteredStocks)


# ==============================================================================
# Cell 9
# ==============================================================================

# Try using this function 
maxBeta = 1.9
maxPE = 40

filteredStocks = filterStocks(maxBeta, maxPE)
print('Accepted filtered stocks are:', filteredStocks)
