# Variables ----------------------------------------------------------------------------------------------------------------------
# Variables are containers for storing data values.

'''
Creating Variables
Python has no command for declaring a variable.

A variable is created the moment you first assign a value to it.
'''
x = 5
y = "Azaz"
print(x)
print(y)


x = 4       # x is of type int
x = "Sally" # x is now of type str
print(x)


'''
Casting
If you want to specify the data type of a variable, this can be done with casting.
'''

x = str(3)    # x will be '3'
y = int(3)    # y will be 3
z = float(3)  # z will be 3.0
print(x,y,z)


'''
Get the Type
You can get the data type of a variable with the type() function.
'''

x = 5
y = "John"
print(type(x))
print(type(y))


'''
Case-Sensitive
Variable names are case-sensitive
'''

a = 4
A = "Sally"
#A will not overwrite a
print(A , a)


# Variable Names---------------------------------------------------------------------------------------------------

# Legal variable names:

myvar = "John"
my_var = "John"
_my_var = "John"
myVar = "John"
MYVAR = "John"
myvar2 = "John"

# llegal variable names:

'''
2myvar = "John"
my-var = "John"
my var = "John"
'''

'''
Multi Words Variable Names

Camel Case
Pascal Case
Snake Case
'''

myVariableName = "md Azaz"
print(myVariableName
)
MyVariableName = "Md Azaz"
print(MyVariableName)

my_variable_name = "md azaz"
print(my_variable_name)



# Many Values to Multiple Variables ----------------------------------------------------------------------------------------------------------

# Python allows you to assign values to multiple variables in one line:

x, y, z = "Orange", "Banana", "Cherry"
print(x)
print(y)
print(z)


x = y = z = "Orange"
print(x)
print(y)
print(z)


'''
Unpack a Collection
If you have a collection of values in a list, tuple etc. Python allows you to extract the values into variables. This is called unpacking.
'''


fruits = ["apple", "banana", "cherry"]
x, y, z = fruits
print(x)
print(y)
print(z)



# Output Variables ---------------------------------------------------------------------------------------------------------------
# The Python print() function is often used to output variables.

x = "Python is awesome"
print(x)



x = "Python"
y = "is"
z = "awesome"
print(x, y, z)



x = "Python "
y = "is "
z = "awesome"
print(x + y + z)



x = 5
y = 10
print(x + y)

# In the print() function, when you try to combine a string and a number with the + operator, Python will give you an error:
'''
x = 5
y = "John"
print(x + y)
'''

x = 5
y = "John"
print(x, y)


# Global Variables -----------------------------------------------------------------------------------------

'''
Variables that are created outside of a function (as in all of the examples above) are known as global variables.

Global variables can be used by everyone, both inside of functions and outside.
'''

x = "awesome"

def myfunc():
  print("Python is " + x)

myfunc()


# -------------------------------------------------------------------------------------------------------------
x = "awesome"

def myfunc():
  x = "fantastic"
  print("Python is " + x)

myfunc()

print("Python is " + x)