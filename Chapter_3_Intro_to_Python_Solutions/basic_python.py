"""
Converted from: Chapter 3 - Exercise 1 - Basic Python.ipynb

This script contains code extracted from a Jupyter notebook.
Code has been organized for modular execution.
"""

import numpy as np # Here numpy is imported as np so I can just type "np." instead of the full "numpy." when calling a function.

"""
Introduction to Python
"""

# Code from Book: Build Your Own AI Investor
# Damon Lee 2021
# Check out the performance on www.valueinvestingai.com
# Code uses data from the (presumably) nice people at https://simfin.com/. 
# Feel free to fork this code for others to see what can be done with it.

# Arithmetic

# Hello! You should know how to execute commands in notebook cells now. Below is some Arithmetic you can try out.

4+3

3-4

4*9

2/5

(2+4)/5*9

# Come up with some yourself and execute them

4+9\
+52\
/5

# This is how you import a module

np.exp(9)

np.log(2)

# Code Comments, Titles and Text

"""
My First Title
"""
# with a little text below the title. Press Shift+Enter to make it proper.

# In the toolbar on the top of this page about here^ should be a drop down menu. Notebooks can have headings and normal text in them if you use that menu to make markdown cells like this one.

# This is also a markdown cell. If you just type in text, it will appear as a paragraph.

# You can also write equations like this: $x^2 + \frac{a}{27}$

'''
This code comment is between triple quotes.
None of this text will be executed as code.

Often this is used for explanations of functions,
or other bodies of text that are best placed
within the code.
'''

"""
The quotes can be single or double quotes.
They are interchangeable in Python,
however they must match to end the quote.
"""

print("hello") # This however will be executed.

# Using Numpy
# Chapter 3 intro to Python functions. Here Numpy is imported for me to use the exponential and logarithmic functions.

# When you import the same module twice, nothing happens the second time.

x = np.exp(9) # make x a number.
print('My value of a is: ', x) #The Print function may take more than one argument to print out things.

y = np.log(x*2) # calculate a value for y
print('My value for y is:', y, ' After computation with x') # Print function might even take three arguments.

print(y)

print('Final answer of computation is:', y)

del y # Variables can be deleted if you so wish.

print(y)

# Strings
# Read through the cells below and execute each cell

my_string = 'This is a string'
print(my_string)

# Strings can be concatenated(added) together
x = 'Further into the unexplored'
y = ' amidst which we live'
print(x+y)

# Special characters in strings are possible
print('\tThis string starts with a tab.')
print('This string \nis split over\na few lines.')

# .format() is a useful feature with Python strings
my_age=32
print('My age is: {}'.format( my_age-2 )   )

# Make up strings for a, b and c
a = "He marched them up "
b = "to the top of the hill "
c = "and he marched them down again."

# Fix the code!
# Print the concatenation of all 3 strings
print(  a + b + c   )
