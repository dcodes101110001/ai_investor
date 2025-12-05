"""
Converted from: Chapter 3 - Exercise 2 - Python Lists and Sets.ipynb

This script contains code extracted from a Jupyter notebook.
Code has been organized for modular execution.
"""

import numpy as np # Here numpy is imported as np so I can just type np. instead of the full numpy. when calling a function.

# Code from Book: Build Your Own AI Investor
# Damon Lee 2021
# Check out the performance on www.valueinvestingai.com
# Code uses data from the (presumably) nice people at https://simfin.com/. 
# Feel free to fork this code for others to see what can be done with it.

"""
Exercise 2 - Python Lists and Sets
"""
# Chapter 2 intro to Python functions. Here Numpy is imported for me to use the exponential and logarithmic functions.
# Lists
# Read through and execute these cells and see what the code does

# Code from Book: Build Your Own AI Investor
# Damon Lee 2021
# Check out the performance on www.valueinvestingai.com
# Code uses data from the (presumably) nice people at https://simfin.com/. 
# Feel free to fork this code for others to see what can be done with it.

a = [1,2,3,4,5] # He's making a list,
print(a)

print(a) # He's checking it twice.

b = [1,2,3,4,6.4, True, False, 'Some Text']

b

c = a+b

print(c)

print('The second item in list c is:', c[1])

print('The tenth item in list c is:', c[9])

c[0:10] # See first 10 items in list

c[0]=200
print(c)

print('The last item in list c is:', c[-1])
print('The second last item in list c is:', c[-2])

my_length = len(c)
print('The length of my list is: ', my_length)

# Stocks list
stocks = ['TSLA', 'BRKA', 'RDSA', 'AMD', 'MMM']
print('Current stocks list: ', stocks)

# 'TSLA' is removed
stocks.remove('TSLA')

# Updated stocks list
print('Updated stocks list: ', stocks)

#Add a new stock to end of list
stocks.append('SAVE')
print('Updated stocks list: ', stocks)

print('My list: ', c) # take a look at the list

# Tuples
# The same as lists, but they are immutable (you can't change them)

my_tuple = (4, 5, 6, 7)
print('My tuple: ', my_tuple)

my_tuple[0] = 5 # Won't work, Tuples are immutable.

# Set Theory
# With Python sets we can do some set theory operations.

my_set = set(c)

print('My set: ', ) # make a set based on the list. Not only unique items appear

#Set A
A = {'Cow', 'Cat', 'Dog', 'Rabbit', 'Sheep'}

#Set B
B = {'Cat', 'Dog', 'Squirrel', 'Fox'}

#Calculating the Union of two sets
U = A.union(B)
print('The union of set A and set B is: ', U)

#Calculating the Intersection of two sets
n = A.intersection(B)
print('The intersection of set A and set B is: ', n)

#Calculating the difference of two sets
d = A.difference(B)
print('The difference of set A and set B is: ', d)

"""
Exercise 2
"""
# Lists
# Some manipulation and accessing variables

# Here's a list of lists.
listOfBestLists = ['NYT best-sellers list', 'Santas list', 'Bucket list', 'Lloyds list', 'Listerine']
# One of these lists is not in fact a list. Either remove it or substitute it with your own list.

# Print out the last item in the list:
print(listOfBestLists[?])

# You can actually print out the second last item by asking for item -2.
print(listOfBestLists[?])

# You know you can access a range with the colon:, this quite versatile as you can leave a number out to slice lists.
first_two_items = listOfBestLists[:2]
print('First Two Items are: {}'.format(first_two_items))

third_to_end = listOfBestLists[2:]
print('First Two Items are: {}'.format(third_to_end))

# You can declare several variables directly form lists in one line.
item1, item2 = listOfBestLists[:2]
print(item1,',', item2)

list_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# Print out the first 5 items
print(list_1[?])

# Print out the last 3 items
print(list_1[?])

# Print out items 3 to 6
print(list_1[?])

# All Set?
# Some code is missing, try fill it out to get it working again

# Here is a list of stocks in my friends portfolio:
friends_stock_list = ['F', 'AMD', 'GAW', 'TM17', 'DISH']

# My friend added MSFT and AMZN stocks to his portfolio yesterday, make a list of those stocks.
stocks_added_yesterday = ?

# Combine both lists together to update his list oh current holdings.
friends_stock_list = friends_stock_list + ?

# He realises the list is incorrect, he sold Ford a sold long ago. correct the list by removing 'F'
friends_stock_list.?

# His sister also holds some stocks, here is the list of them:
sisters_stocks_list = ['DIS', 'TM17', 'MMM', 'AMD']

# We want to see what stocks they have in common (I know it's not many, imagine they contain 40+ each). 
# for each list create a corresponding set, and find their intersection.
friends_stock_set = ?
sisters_stocks_set = ?

#print('Their common holdings are: ', friends_stock_set.?(sisters_stocks_set))

# Let's see a complete set of all their holdings.
print('Their total holdings are: ', friends_stock_set.?(sisters_stocks_set))
