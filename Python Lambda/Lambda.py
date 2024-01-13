#  Lambda -------------------------------------------------------------------------------------------------------------------------------------------------------

'''
A lambda function is a small anonymous function.

A lambda function can take any number of arguments, but can only have one expression.
'''
# Example
# Add 10 to argument a, and return the result:

x = lambda a : a + 10
print(x(5))

# Lambda functions can take any number of arguments:

'''
Example
Multiply argument a with argument b and return the result:
'''

multi = lambda a,b : a*b
print(multi(5,6))


'''
Example
Summarize argument a, b, and c and return the result:

'''

sum = lambda a, b, c : a + b + c
print(sum(5, 6, 2))


'''
Why Use Lambda Functions?
The power of lambda is better shown when you use them as an anonymous function inside another function.

Say you have a function definition that takes one argument, and that argument will be multiplied with an unknown number:
'''

def my_function (n):
    return lambda a : a*n

my_dou = my_function(2)


print(my_dou(2))

# -----------------------------------------------------------------------------------------------------------------------

def myfunc(n):
  return lambda a : a * n

mytripler = myfunc(3)

print(mytripler(11))


#  Or, use the same function definition to make both functions, in the same program:

def myfun(n):
    return lambda m : m*n

my_main1 = myfun(2)
my_main2 = myfun(2)

print(my_main1(5))
print(my_main1(5))
