# Python Functions -----------------------------------------------------------------------------------------------------------------------------

'''
A function is a block of code which only runs when it is called.

You can pass data, known as parameters, into a function.

A function can return data as a result.
'''

'''
Creating a Function ----------------------------------------------------------------------------------------------------------------------------
In Python a function is defined using the def keyword:
'''

def my_function():
    print("Hello from a function")


def myfun():
  print(f'hello world')
  
myfun()






'''
Calling a Function ------------------------------------------------------------------------------------------------------------------------------
To call a function, use the function name followed by parenthesis:
'''

def my_function():
  print("Hello from a function")

my_function()



'''
Arguments ----------------------------------------------------------------------------------------------------------------------------------------
Information can be passed into functions as arguments.

Arguments are specified after the function name, inside the parentheses. You can add as many 
arguments as you want, just separate them with a comma.The following example has a function with
one argument (fname). When the function is called, we pass along a first name, which is used
inside the function to print the full name:
'''

def my_function(fname):
  print(fname + " Refsnes")

my_function("Azaz")
my_function("Lux")
my_function("Harshit")



'''
Parameters or Arguments? ---------------------------------------------------------------------------------------------------------------------------
The terms parameter and argument can be used for the same thing: information that are passed into a function.

From a function's perspective:

A parameter is the variable listed inside the parentheses in the function definition.

An argument is the value that is sent to the function when it is called.
'''


'''
Number of Arguments -----------------------------------------------------------------------------------------------------------------------------------
By default, a function must be called with the correct number of arguments. Meaning that if your function
expects 2 arguments, you have to call the function with 2 arguments, not more, and not less.
'''

# Example
# This function expects 2 arguments, and gets 2 arguments:

def my_function(fname, lname):
  print(fname + " " + lname)

my_function("Md", "Azaz")

# If you try to call the function with 1 or 3 arguments, you will get an error:

# Example
# This function expects 2 arguments, but gets only 1:

def my_function(fname, lname):
  print(fname + " " + lname)

# my_function("Emil")


def add(a, b):
  sum = a+b
  return sum


print(add(2,2))




'''
Arbitrary Arguments, *args --------------------------------------------------------------------------------------------------------------------------------
If you do not know how many arguments that will be passed into your function, add a * before the parameter name in the function definition.

This way the function will receive a tuple of arguments, and can access the items accordingly:
'''

def my_function(*name):
  print("My Name is " + name[0])

my_function("Md", "Azaz")

# ------------------------------------------------------------------------

def my_function(*kids):
  print("The youngest child is " + kids[1])

my_function("Babu", "Lux", "Harshit")

# Arbitrary Arguments are often shortened to *args in Python documentations.



'''
Keyword Arguments -----------------------------------------------------------------------------------------------------------------------------------------------
You can also send arguments with the key = value syntax.

This way the order of the arguments does not matter.
'''

def my_function(child3, child2, child1):
  print("The youngest child is " + child3 + "," + child2 )

my_function(child1 = "Babu", child2 = "Lux", child3 = "Harshit")



'''
Arbitrary Keyword Arguments, **kwargs ---------------------------------------------------------------------------------------------------------------------------
If you do not know how many keyword arguments that will be passed into your function, add two asterisk: ** before the parameter name in the function definition.

This way the function will receive a dictionary of arguments, and can access the items accordingly:
'''

def my_function(**kid):
  print("His last name is " + kid["lname"])

my_function(fname = "Md", lname = "Azaz")


'''
Default Parameter Value ----------------------------------------------------------------------------------------------------------------------------------------
The following example shows how to use a default parameter value.

If we call the function without argument, it uses the default value:
'''

def my_function(country = "Norway"):
  print("I am from " + country)

my_function("Sweden")
my_function("India")
my_function()
my_function("Brazil")



'''
Passing a List as an Argument ----------------------------------------------------------------------------------------------------------------------------------
You can send any data types of argument to a function (string, number, list, dictionary etc.), and it will be treated as the same data type inside the function.

E.g. if you send a List as an argument, it will still be a List when it reaches the function:
'''
def my_function(food):
    for x in food:
        print(x)

fruits = ["apple", "banana", "cherry"]
my_function(fruits)




'''
Return Values --------------------------------------------------------------------------------------------------------------------------------------------------
To let a function return a value, use the return statement:
'''

def my_function(x):
  return 5 * x

print(my_function(3))
print(my_function(5))
print(my_function(9))


'''
The pass Statement --------------------------------------------------------------------------------------------------------------------------------------------
function definitions cannot be empty, but if you for some reason have a function 
definition with no content, put in the pass statement to avoid getting an error.
'''

def myfunction():
  pass




'''
Positional-Only Arguments -------------------------------------------------------------------------------------------------------------------------------------
You can specify that a function can have ONLY positional arguments, or ONLY keyword arguments.

To specify that a function can have only positional arguments, add , / after the arguments:
'''


def my_function(x, /):
  print(x)

my_function(3)

# -----------------------------------------------

def my_function(x):
  print(x)

my_function(x = 3)

# -----------------------------------------------

def my_function(x, /):
  print(x)

# my_function(x = 3)


'''
Keyword-Only Arguments -----------------------------------------------------------------------------------------------------------------------------------------
To specify that a function can have only keyword arguments, add *, before the arguments:

Example
'''


def my_function(*, x):
  print(x)

my_function(x = 3)



'''
Combine Positional-Only and Keyword-Only -----------------------------------------------------------------------------------------------------------------------
You can combine the two argument types in the same function.

Any argument before the / , are positional-only, and any argument after the *, are keyword-only.

Example
'''

def my_function(a, b, /, *, c, d):
  print(a + b + c + d)

my_function(5, 6, c = 7, d = 8)


'''
Recursion------------------------------------------------------------------------------------------------------------------------------------------------------
Python also accepts function recursion, which means a defined function can call itself.
'''

def tri_recursion(k):
  if(k > 0):
    result = k + tri_recursion(k - 1)
    print(result)
  else:
    result = 0
  return result

print("\n\nRecursion Example Results")
tri_recursion(6)


# Global Variable / Scope --------------------------------------------------------------------------------------------------------------------------------------


x = 101 # Global Scope


def func():
  x = 102 # Local Scope
  print(x)

func()
print(x)


# Default Argument --------------------------------------------------------------------------------------------------------------------------------------

def greet(name , message = 'good morning'): # Default Argument : - {message = 'good morning'}
  print(f'Hi my name was {name} and {message} and Lux')


greet('Azaz')


# keyword Argument

def greet(name , age , message):
  print(f'{message} {name} your age is {age}')

greet(name='Azaz' , age=22 , message='Hello')



# *args Argument  ---------------------------------------------------------------------------------------------------------------------------------------

def sum_num(*args):
  print(type(args))
  print(args)

  sum = 0
  for i in args:
    sum+=i
  return sum



print(sum_num(1,2,3,4,5))



# **kwargs Argument  ---------------------------------------------------------------------------------------------------------------------------------------


def info(**kwargs):
  print(type(kwargs))
  print(kwargs)

  for key , value in kwargs.items():
    print(f'{key} --> {value} ')




info(name = 'Azaz' , age = 22 , city = 'Kanpur')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------


def func1(a,b,*args , **kwargs):
  print(a)
  print(b)
  print(args)
  print(kwargs)

func1(1,2,3,4,5, name = 'Azaz', age=22)


# -------------------------------------------------------------------


def sums(a=int , b=int)->int:
  print(a+b)


sums(2,2)


# -------------------------------------------------------------------------

def fun_outer():
  print('hello outer')

  def fun_inner():
    print('hello inner')
    return(fun_outer)




fun_outer()()
