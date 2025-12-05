"""
Converted from: Chapter 3 - Exercise 5 - Numpy Basics.ipynb

This script contains code extracted from a Jupyter notebook.
Code has been organized for modular execution.
"""

import matplotlib.pyplot as plt
import numpy as np # Import Numpy
from matplotlib import pyplot as plt

"""
Exercise 5 – Numpy Basics
"""
# Here we get aquainted with Numpy.

# Code from Book: Build Your Own AI Investor
# Damon Lee 2021
# Check out the performance on www.valueinvestingai.com
# Code uses data from the (presumably) nice people at https://simfin.com/. 
# Feel free to fork this code for others to see what can be done with it.

# Set the plotting DPI settings to be a bit higher.
plt.rcParams['figure.figsize'] = [8.0, 5.0]
plt.rcParams['figure.dpi'] = 150

# Some basics

# The library has mathematical functions like in the math library, for example:
print(np.log(9)) # Logarithm
print(np.exp(9)) # Exponential
print(np.sin(9)) # Sine
print(np.cos(9)) # Cosine

# However the main use of Numpy is to do computing with matricies.
# To make a 1 dimensional array use np.array and pass a list to it:
# (It is like a list except all items in the array ARE ALWAYS 
# the same type. e.g. strings, integers, floating point numbers)
a = np.array([1, 2, 3, 4, 5])
print(a)

# Can also make an array with range().
a = np.array(range(5,10))
print('\nAn array made with range() instead:', a)

# For evenly spaced numbers over a specified interval use linspace
a = np.linspace(1.2, 3.8, num=5)
print('\nAn array made with linspace() instead:', a)

# To see the type of object this is:
print('\nnumpy array type is: ', type(a))

# As with lists you can see individual items like so:
print('\nThe first item is:', a[0])
print('The second item is:', a[1])
print('The third item is:', a[2])

# You can change items in the array the same was as with lists:
a[0] = 9 # change first number to be 9.
print('\nThe first item is now:', a[0])

# As with lists, you can select a range of values with numpy:
a[0:3] = 99 # change the first 3 items to be 999
print('\nThe array is now:', a)

# we can find max, min, mean etc.
print('\nThe max of our array is:', a.max())
print('The min of our array is:', a.min())
print('The mean of our array is:', a.mean())

# Numpy arrays can have more than one dimension, 
# for example the following is a 2D matrix, 
# created with a list of lists.
m = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]) 
print(m)

# You can extract the size in any dimension from 
# this array with .shape (Which returns a tuple)
print('\nThe matrix dimensions are:', m.shape)

# With multidimensional arrays you can see and 
# change items within them with indexing, but you 
# need as many indexes as there are dimensions to 
# specify the item location.
# Remember indexing begins at 0!
print('\nItem in first row, third column is: ', m[0,2])
print('Item in second row, fifth column is: ', m[0,4])

# Arrays can be manipulated in various ways, for instance rehape:
rsm = m.reshape(5,2) # Change the shape to be 5 rows and 2 columns
print('\nReshaped matrix is: \n', rsm)

tm = m.transpose()
print('\nTransposed matrix is: \n', tm)

into_line = m.ravel()
print('\nUnraveled matrix is: \n', into_line)

# You can also create numpy arrays with zeros, ones in them etc.
a_z = np.zeros((3,4)) # Create an array with just zeros in it, 3x4
print('\n', a_z)             

a_o = np.ones((10,8)) # Create an array with just ones in it
print('\n', a_o)

a_pi = np.full((5,5), 3.14) # Create an array filled with one number
print('\n', a_pi)

# Remember these arrays are multidimensional, 
# you can make a 3D array if you really need it...
d_mat = np.ones((3,3,3))
print('\n Here is a 3D matrix: \n', d_mat)

# Array indexing works just the same with 2D matrices, 
# for instance with our array of ones:
a_o[2:6,2:]=9.87
print('Our large matrix of ones is now: \n', a_o)

# numpy functions can be used on whole arrays, say if
# we want to do an exponential on every value in the matrix:
a_o_exp = np.exp(a_o)
print('Our large matrix is now: \n', a_o_exp)

# Arrays can be added to each other, divided by each other etc.
arr_1 = np.array([[1,2,3], [4,5,6], [7,8,9]])
arr_2 = np.array([[7,6,4], [12,5,1], [2,2,249]])

print('Array 1:\n', arr_1)
print('\nArray 2:\n', arr_2)

# Do some calculations on arrays
arr_3 = arr_2 + np.sin(arr_1) - arr_1/arr_2 
print('\nArray 3:\n', arr_3)

# Remember arrays need to be the appropriate size to do operation between them.

# Running functions on Numpy arrays in a loop

# It is often useful to iterate over an array, 
# lets make an array and iterate over it.
a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
for x in a:
    print(x**2 + x)

# Plotting Numpy arrays

#Import a plotting library for Python to plot some numpy arrays

x = np.array(range(0,9))
y = np.sqrt(x) # run a numpy function on the entire array.
plt.plot(x,y)
print('\nOur array y has values: ', y)

"""
Discounted Cash Flow Bare Bones
"""

numberYears = 20 # Number of years to project out
yearNumber = np.array(range(1,numberYears+1)) # An array tracking the cash flow years [1, 2, 3, 4,...
CF = np.ones(numberYears) * 1000 # Assumption of cash flow of $1000 per year

plt.bar(yearNumber, height=CF)
plt.ylabel('Cash Flow Value $')
plt.xlabel('Cash Flow Year')

DCF = np.array([]) # Empty array for DCF to be populated
discountRate = 0.16 # 0.1 for 10%, 0.15 for 15%...

# Loop through the cash flows and populate an array with equivalent 
# disocunted cash flows
for i in range(0, len(CF)):
    
    # Discounted cash flow for year i
    discountCashFlow = CF[i]/pow((1+discountRate),(i+1)) 
    
    # Print out the values to see the calculation
    print('Year '+str(i)+'\tOriginal Cash Flow: ', CF[i], 
          '\t', 'Discounted Cash Flow: ', round(discountCashFlow,1))
    
    # Append to the DCF array for plotting later
    DCF = np.append(DCF, discountCashFlow)

plt.bar(yearNumber, height=CF)
plt.bar(yearNumber, height=DCF)
plt.ylabel('Cash Flow Value $')
plt.xlabel('Cash Flow Year')
plt.legend(['Cash Flow', 'Discounted Cash Flow'])

DCF.sum() # Present value of discounted future cash flows

# Excercise 5 Real Company DCF
# Here we will do a discounted cash flow calculation with Numpy.
#
# Unfortunately Toby has messed up our code! Try and fix it and get it working. Anything with a '?' has to be rectified.

# Nice please to see cash flows is https://www.macrotrends.net/stocks/charts/INTC/intel/free-cash-flow

#Here is the rough cash flow data for the company. Values in millions
cashFlow_years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
freeCashFlow = [10313.00, 11692.00, 12183.00, 10332.00, 14251.00, 16932.00, 20931.00]#intc
freeCashFlowData = np.array([cashFlow_years, freeCashFlow])

growth = np.zeros(6)
for i in range(1, len(growth)+1):
    growth[i-1] = (freeCashFlowData[1,i]/freeCashFlowData[1,i-1]-1) * 100

def calcDCF(discRate, growthRate, currentFcf, numYears, startYear):
    '''
    Define a function that returns an array containing cash flow and 
    discounted cash flow for each future year.
    '''
    
    # Start zero arrays and fill the array with a loop.
    futureCashF???s = np.zeros(numY????)
    futureDiscCashFlows = np.???os(numYears)
    
    # Now fill the array with a loop
    for i in ra???(0, numYears):
        futureCashFlows[i] = currentFcf * (growthRate+1) ** i
        futureDiscCashFlows[i] = futureCashFlows[i]/((1+discRate)**i)
    
    # Have an array for the future years, from say, 2020 to 2020 + numYears
    futureYears = np.ar???(ra???(startYear, startYear+numYears))
    
    # Return the arrays of: future cash flows
    # Future discounted Cash Flows
    # Future years that those cash flowws occur in
    return [futureC???Flows, fut???DiscCashFlows, futureY????]

# Run the function
# Feel free to change the function arguments and see how they impact the valuation
[futureCashFlows, futureDiscCashFlows, futureYears] = calcDCF(discRate=0.???, 
                                                              growthRate=0.05, 
                                                              currentFcf=19000, 
                                                              numYears=???, 
                                                              startYear=2020)

# plot the past cash flows, future cash flows, 
# and discounted future cash flows on a graph.
plt.plot(freeCashFlowData[0,:], 
         freeCashFlowData[1,:], 
         '-s', 
         label='Past free cash flows')

plt.plot(futureYears, futureCashFlows, 
         '--d', label='Projected future cash flows')

plt.plot(futureYears, futureDiscCashFlows, 
         '-x', label='Discounted value of future cash flows')


plt.fill_between(futureYears, futureDiscCashFlows, 0, alpha=0.3)
#plt.ylim([0, 35])
plt.legend()
plt.grid()
plt.ylabel('INTC Free Cash Flow (For Year)')
plt.title('Discounted Cash Flow Projection For INTC')

# Print out the net present value of future cash flows.
print('The future value of discounted cash flows(the estimated company value) is:\n',
      round(futureDiscCashFlows.sum()/1000, 2),
      'Billion dollars')
