#!/usr/bin/env python3
# Converted from: Chapter 3 - Exercise 5 - Numpy Basics.ipynb


# ==============================================================================
# Cell 1
# ==============================================================================
# # Exercise 5 – Numpy Basics
# Here we get aquainted with Numpy.


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

# Set the plotting DPI settings to be a bit higher.
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8.0, 5.0]
plt.rcParams['figure.dpi'] = 150


# ==============================================================================
# Cell 4
# ==============================================================================
# # Exercise 5
# Here we will do a discounted cash flow calculation with Numpy.
#
# #### Unfortunately Toby has messed up our code! Try and fix it and get it working.


# ==============================================================================
# Cell 5
# ==============================================================================

import numpy as np
from matplotlib import pyplot as plt


# ==============================================================================
# Cell 6
# ==============================================================================

import numpy as np
from matplotlib import pyplot as plt

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
    futureCashFlows = np.zeros(numYears)
    futureDiscCashFlows = np.zeros(numYears)
    
    # Now fill the array with a loop
    for i in range(0, numYears):
        futureCashFlows[i] = currentFcf * (growthRate+1) ** i
        futureDiscCashFlows[i] = futureCashFlows[i]/((1+discRate)**i)
    
    # Have an array for the future years, from say, 2020 to 2020 + numYears
    futureYears = np.array(range(startYear, startYear+numYears))
    
    # Return the arrays of: future cash flows
    # Future discounted Cash Flows
    # Future years that those cash flowws occur in
    return [futureCashFlows, futureDiscCashFlows, futureYears]




# Run the function
# Feel free to change the function arguments and see how they impact the valuation
[futureCashFlows, futureDiscCashFlows, futureYears] = calcDCF(discRate=0.15, 
                                                              growthRate=0.05, 
                                                              currentFcf=19000, 
                                                              numYears=20, 
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


# ==============================================================================
# Cell 7
# ==============================================================================

# Print out the net present value of future cash flows.
print('The future value of discounted cash flows(the estimated company value) is:\n',
      round(futureDiscCashFlows.sum()/1000, 2),
      'Billion dollars')
