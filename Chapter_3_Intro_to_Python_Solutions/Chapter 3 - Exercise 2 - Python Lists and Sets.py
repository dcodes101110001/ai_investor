#!/usr/bin/env python3
# Converted from: Chapter 3 - Exercise 2 - Python Lists and Sets.ipynb


# ==============================================================================
# Cell 1
# ==============================================================================
# # Exercise 2 - Python Lists and Sets
# Chapter 2 intro to Python functions. Here Numpy is imported for me to use the exponential and logarithmic functions.
# ## Lists
# Read through and execute these cells and see what the code does


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
# # Exercise 2
# ## Lists
# Some manipulation and accessing variables


# ==============================================================================
# Cell 4
# ==============================================================================

# Here's a list of lists.
listOfBestLists = ['NYT best-sellers list', 'Santas list', 'Bucket list', 'Lloyds list', 'Listerine']
# One of these lists is not in fact a list. Either remove it or substitute it with your own list.


# ==============================================================================
# Cell 5
# ==============================================================================

# Print out the last item in the list:
print(listOfBestLists[-1])


# ==============================================================================
# Cell 6
# ==============================================================================

# You can actually print out the second last item by asking for item -2.
print(listOfBestLists[-2])


# ==============================================================================
# Cell 7
# ==============================================================================

# You know you can access a range with the colon:, this quite versatile as you can leave a number out to slice lists.
first_two_items = listOfBestLists[:2]
print('First Two Items are: {}'.format(first_two_items))


# ==============================================================================
# Cell 8
# ==============================================================================

third_to_end = listOfBestLists[2:]
print('First Two Items are: {}'.format(third_to_end))


# ==============================================================================
# Cell 9
# ==============================================================================

# You can declare several variables directly form lists in one line.
item1, item2 = listOfBestLists[:2]
print(item1,',', item2)


# ==============================================================================
# Cell 10
# ==============================================================================

list_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# Print out the first 5 items
print(list_1[0:5])


# ==============================================================================
# Cell 11
# ==============================================================================

# Print out the last 3 items
last_index = len(list_1)
print(list_1[last_index-3:last_index])


# ==============================================================================
# Cell 12
# ==============================================================================

# Print out items 3 to 6
print(list_1[2:6])


# ==============================================================================
# Cell 13
# ==============================================================================
# ## All Set?
# Some code is missing, try fill it out to get it working again


# ==============================================================================
# Cell 14
# ==============================================================================

# Here is a list of stocks in my friends portfolio:
friends_stock_list = ['F', 'AMD', 'GAW', 'TM17', 'DISH']


# ==============================================================================
# Cell 15
# ==============================================================================

# My friend added MSFT and AMZN stocks to his portfolio yesterday, make a list of those stocks.
stocks_added_yesterday = ['MSFT', 'AMZN']


# ==============================================================================
# Cell 16
# ==============================================================================

# Combine both lists together to update his list oh current holdings.
friends_stock_list = friends_stock_list + stocks_added_yesterday


# ==============================================================================
# Cell 17
# ==============================================================================

# He realises the list is incorrect, he sold Ford a sold long ago. correct the list by removing 'F'
friends_stock_list.remove('F')


# ==============================================================================
# Cell 18
# ==============================================================================

# His sister also holds some stocks, here is the list of them:
sisters_stocks_list = ['DIS', 'TM17', 'MMM', 'AMD']


# ==============================================================================
# Cell 19
# ==============================================================================

# We want to see what stocks they have in common (I know it's not many, imagine they contain 40+ each). 
# for each list create a corresponding set, and find their intersection.
friends_stock_set = set(friends_stock_list)
sisters_stocks_set = set(sisters_stocks_list)


# ==============================================================================
# Cell 20
# ==============================================================================

print('Their common holdings are: ', friends_stock_set.intersection(sisters_stocks_set))


# ==============================================================================
# Cell 21
# ==============================================================================

# Let's see a complete set of all their holdings.
print('Their total holdings are: ', friends_stock_set.union(sisters_stocks_set))
