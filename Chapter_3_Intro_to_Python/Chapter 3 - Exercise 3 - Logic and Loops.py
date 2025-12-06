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
# ## Custom Functions
# ### You can make your own functions.


# ==============================================================================
# Cell 4
# ==============================================================================

# Here is a simple fuction. All it does is say hello.
def sayHello():
    print("Hello!")


# ==============================================================================
# Cell 5
# ==============================================================================

#let's call the function
sayHello()


# ==============================================================================
# Cell 6
# ==============================================================================
# ### You can pass arguments into functions


# ==============================================================================
# Cell 7
# ==============================================================================

# You can pass values to functions (called arguments)
def functionWithArg(arg_1):
    print(arg_1)


# ==============================================================================
# Cell 8
# ==============================================================================

functionWithArg('Print stuff.')
functionWithArg('ABC 123 XYZ. This is a single string BTW.')


# ==============================================================================
# Cell 9
# ==============================================================================

# You can pass many arguments into your function, 
# here 3 arguments are used.
def addThree(first, second, third):
    answer = first + second + third
    print('The sum result is:', answer)
    
    return answer


# ==============================================================================
# Cell 10
# ==============================================================================

mySum = addThree(9,200,12)


# ==============================================================================
# Cell 11
# ==============================================================================

print(mySum)


# ==============================================================================
# Cell 12
# ==============================================================================

def multiplyThree(first, second, third):
    '''
    It is good practice to put code comments here
    To explain what the function does, for anyone who
    may use it in future.
    
    This function multiplies three input arguments together,
    and prints the result. The result is returned too.
    '''
    answer = first * second * third
    print('The multiplicative result is:', answer)
    
    return answer

myMultAns = multiplyThree(4,3,2)


# ==============================================================================
# Cell 13
# ==============================================================================
# ### You can pass many varied arguments at once


# ==============================================================================
# Cell 14
# ==============================================================================

# You can also pass many different kinds of objects
# as arguments to functions
# Here we pass a list and a number into an argument.

def retFirstNStocks(stockList, n):
    return stockList[:n] # returns all items up to index n

# Making a list to pass into the function.
friends_stock_list = ['F', 'AMD', 'GAW', 'TM17', 'DISH', 'INTC']

# Now we assign a new variable to what the function returns.
reduced_list = retFirstNStocks(friends_stock_list, 3) 

#lets see what the returned variable is:
print('The object our function returned to us is:', reduced_list)
print('\n This object is a:', type(reduced_list))


# ==============================================================================
# Cell 15
# ==============================================================================
# ### Functions can return a list of several things at once


# ==============================================================================
# Cell 16
# ==============================================================================

# A function can return many things in the 'return' line with a list.
def getDataFromList(stockList):
    '''
    This returns a list of:
    First 3 items in list
    first item
    last item
    number of items
    '''
    size = len(friends_stock_list)
    
    # remember, indexing starts at 0
    print('The first item is:', stockList[0])
    print('The last item is:', stockList[size-1]) 
    print('The number of items is:', size)
    
    return [stockList[:3], stockList[0], stockList[size-1], size]


# ==============================================================================
# Cell 17
# ==============================================================================

# here we get all the results from the function in the variable 'list_results'
list_results = getDataFromList(friends_stock_list)

#Taking a look at this variable, there are a few kinds of objects there all in one list:
print('\n',list_results)


# ==============================================================================
# Cell 18
# ==============================================================================
# ### Arguments can be optional


# ==============================================================================
# Cell 19
# ==============================================================================

# Making the argument optional
def printMyNumber(num='I would like to have an argument please.'):
    print('My number is:', num)
    
    
printMyNumber(3.141) # With an argument
printMyNumber() # Without an argument, reverting to your default


# ==============================================================================
# Cell 20
# ==============================================================================
# # Logic and Loops


# ==============================================================================
# Cell 21
# ==============================================================================
# ### Basic FOR loops


# ==============================================================================
# Cell 22
# ==============================================================================

friends_stock_list = ['F', 'AMD', 'GAW', 'TM17', 'DISH', 'INTC']

#Here is a loop over friends_stock_list
for item in friends_stock_list:
    print(item)


# ==============================================================================
# Cell 23
# ==============================================================================

#Here is a loop over a range beginning at 0 and ending at 3
for i in range(0,4):
    print(i, 'spam')


# ==============================================================================
# Cell 24
# ==============================================================================

cube_numbers = [x**3 for x in range(10)]
print(cube_numbers)


# ==============================================================================
# Cell 25
# ==============================================================================
# ### Boolean types and some logic


# ==============================================================================
# Cell 26
# ==============================================================================

# Boolean
num = 99 # here we have a variable

num == 4 # the == operator will return a boolean here, which is either True of False.


# ==============================================================================
# Cell 27
# ==============================================================================

#lets see what it returns when we ask if it is 99.
num == 99


# ==============================================================================
# Cell 28
# ==============================================================================

# Greater than and les than operators are also available to us.

print('Is num greater than 100?', (num > 100) ) # (num>100) returns a bool.

print('Is num Less than 100?', (num < 100) ) # (num<100) returns a bool.


# ==============================================================================
# Cell 29
# ==============================================================================

num2 = 100 # create a second number

bool1 = (num < 100)
print('boolean 1 is:',bool1)

bool2 = (num2 < 100)
print('boolean 2 is:',bool2)

bool3 = bool1 | bool2
print('\nboolean 3 (which is bool1 or bool2 being True) is:\n',bool3)


# ==============================================================================
# Cell 30
# ==============================================================================
# ### Table of Operators
# | Basic Operators | Description |
# | :-: | :-: |
# | < | Less than |
# | <= | Less than or equal to |
# | >	| Greater than |
# | >= | Greater than or equal to |
# | == | Equal to |
# | != | Not equal to |
# | &	| And |
# | \|	| Or |


# ==============================================================================
# Cell 31
# ==============================================================================
# ### Some logic in a FOR loop


# ==============================================================================
# Cell 32
# ==============================================================================

for i in [0,1,2,3,4,5,6,7,8,9,12,17,200]:
    boolIsGreaterThanSix = (i >= 6)
    print(i, boolIsGreaterThanSix)


# ==============================================================================
# Cell 33
# ==============================================================================

#Here is a loop over a range, where a logic IF statement is present 
# to only print unmbers that are a multiple of 3
for i in range(0, 30):
    
    # The % is the modulo function, 
    # it returns the remainder of a division.
    my_rem = (i % 3)
    
    if (my_rem == 0): 
        print(i)


# ==============================================================================
# Cell 34
# ==============================================================================

#Multiple of 3 greater than 20
for i in range(0,40):
    
    # Firstly a Boolean,
    # is True if i is greater than 20.
    greaterThan20 = (i > 20) 
    
    # Secondly a Number, 
    # is the remainder of division.
    myRemainder = (i % 3) 
    
    # Thirdly another Bool,
    # is True if i is multiple of 3.
    multOf3 = (myRemainder == 0)
    
    # If a multiple of 3 AND greater than 20.
    if (multOf3 & greaterThan20): 
        print(i)


# ==============================================================================
# Cell 35
# ==============================================================================

def mult3AndGr20(number):
    '''
    Returns True if a multiple of 3 greater than 20.
    Otherwise returns False.
    '''
    # Firstly a Boolean,
    # is True if i is greater than 20.
    greaterThan20 = (i > 20) 
    
    # Secondly a Number, 
    # is the remainder of division.
    myRemainder = (i % 3) 
    
    # Thirdly another Bool,
    # is True if i is multiple of 3.
    multOf3 = (myRemainder == 0)
    
    # If a multiple of 3 AND greater than 20.
    if (multOf3 & greaterThan20): 
        return True
    else:
        return False
    

#Multiple of 3 greater than 20
for i in range(0,40):
    if(mult3AndGr20(i)):
        print(i)


# ==============================================================================
# Cell 36
# ==============================================================================
# # Exercise 3


# ==============================================================================
# Cell 37
# ==============================================================================

# Output stocks with a price/earnings ratio below a number and a beta above a number. 
# return a list of aceptable stocks with ascending by P/E
# Try and separate out these steps and work on one at a time, life is easier that way

# Toby has come and messed up the code! Try and get it working again by fixing the code missing where "?" appears
# Answers are in the answer notebook.


# ==============================================================================
# Cell 38
# ==============================================================================

# Given data.
Stocks = ['JKHY', 'FSTR', 'ETV', 'WAFD', 'RSG', 'HTGC']
PriceEarningsRatios = [48.54, 3.29, 5.71, 9.18, 23.89, 14.02]
Beta = [0.59, 1.67, 0.89, 0.99, 0.62, 1.51]


# ==============================================================================
# Cell 39
# ==============================================================================

# Fix this function
def filterStocks(maxBeta, maxPE):
    '''
    This function reads global list variables:
    Stocks
    PriceEarningsRatios
    Beta
    
    and returns a list of stocks that are below the maxBeta and maxPE values
    '''
    
    filteredStocks = ? # Create an empty list to be appended to.
    
    for i, stock in enumerate(Stocks): # This counts i as 1, 2, 3, ... down the list of stocks
        PriceEarningsBool = (PriceEarningsRatios[i] < ?) # returns True if our current PE is less than maxPE
        
        ? = (Beta[i] < maxBeta) # returns True if our current beta is less than maxbeta
        
        if (? & BetaBool):
            filteredStocks.?(Stocks[i]) # with each loop, if conditions are met, append to the list.
    return ? # Return the filtered Stocks


# ==============================================================================
# Cell 40
# ==============================================================================

# Try using this function
maxBeta = 1.2 # Can try a bunch of numbers
maxPE = 20 # Can try a bunch of numbers

filteredStocks = filterStocks(maxBeta, maxPE)
print('Accepted filtered stocks are:', ?)
