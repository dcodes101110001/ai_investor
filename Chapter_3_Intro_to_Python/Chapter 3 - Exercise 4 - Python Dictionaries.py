#!/usr/bin/env python3
# Converted from: Chapter 3 - Exercise 4 - Python Dictionaries.ipynb


# ==============================================================================
# Cell 1
# ==============================================================================
# # Exercise 4 - Python Dictionaries


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

# A python Dictionary is a kind of lookup table. It can store a series of any kind of object which is accessed with a 'key'.
dict1 = {} # This is how to create en empty dictionary.

# Here we fill our Python dictionary which stores data about one person.
dict1 = {
    'Name': 'Max',
    'Height': 1.9,
    'Test Scores':[50, 62, 78, 47],
    'Nationality':'German'
}
print(dict1)


# ==============================================================================
# Cell 4
# ==============================================================================

# The first item is the 'Key', the item after the colon: is the infomation to be accessed with the key.
dict1['Name'] # View the info behind item 'Name'


# ==============================================================================
# Cell 5
# ==============================================================================

dict1['Height'] # View the info behind item 'Height'


# ==============================================================================
# Cell 6
# ==============================================================================

# You can view all the keys in your dictionary with the .keys() function
dict1.keys()


# ==============================================================================
# Cell 7
# ==============================================================================

# You can add items to Dictionaries by stating them:
dict1['Shoe Size'] = 7
print(dict1)


# ==============================================================================
# Cell 8
# ==============================================================================

# You can delete items with the del command as with all other variables in Python.
del dict1['Height']
print(dict1)


# ==============================================================================
# Cell 9
# ==============================================================================
# # Exercise 4
# Dictionaries are quite a fundamental must-know object in Python, they are essentially look-up tables.


# ==============================================================================
# Cell 10
# ==============================================================================

# Use a loop to fill in a list of dictionaries with company data,
# where each dictionary contains infomation for one stock.
# Stock infomation is provided


# ==============================================================================
# Cell 11
# ==============================================================================

# Once the data is stored in a list of dicts, use a loop and an if statement to filter out our stocks.
# Have the loop create a new list of dictionaries of companies we have filtered.


# ==============================================================================
# Cell 12
# ==============================================================================

# Given data.
Stocks = ['JKHY', 'FSTR', 'ETV', 'WAFD', 'RSG', 'HTGC']
PriceEarningsRatios = [48.54, 3.29, 5.71, 9.18, 23.89, 14.02]
Beta = [0.59, 1.67, 0.89, 0.99, 0.62, 1.51]


# ==============================================================================
# Cell 13
# ==============================================================================

# Correct this loop to fill our list of dictionaries
stockDictDicts = {}#Create empty dict we will fill with infomation
for i, S in enumerate(Stocks): 
    stockDictDicts[S] =  {'PE':PriceEarningsRatios[i],\
                         'Beta':Beta[i]}


# ==============================================================================
# Cell 14
# ==============================================================================

stockDictDicts # View out list of dictionaries with all data


# ==============================================================================
# Cell 15
# ==============================================================================

# With infomation stored in this way we can see infomation about a single company:
stockDictDicts['WAFD']


# ==============================================================================
# Cell 16
# ==============================================================================

# And with a loop we can filter stocks by their internal information:
filt_dict = [] # empty list to fill up
for dict_i in stockDictDicts:
    if(stockDictDicts[dict_i]['PE'] < 10):
        filt_dict.append(dict_i)

print(filt_dict)
