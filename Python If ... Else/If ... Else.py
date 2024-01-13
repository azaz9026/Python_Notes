# Python If ... Else ----------------------------------------------------------------------------------------------------------------------------------------------

'''
Python Conditions and If statements -----------------------------------------------------------------------

Python supports the usual logical conditions from mathematics:

Equals: a == b
Not Equals: a != b
Less than: a < b
Less than or equal to: a <= b
Greater than: a > b
Greater than or equal to: a >= b
These conditions can be used in several ways, most commonly in "if statements" and loops.

An "if statement" is written by using the if keyword.
'''
a = 33
b = 200
if b > a:
  print("b is greater than a")


# Example
# If statement, without indentation (will raise an error):

a = 33
b = 200
if b > a:
  print("b is greater than a") # you will get an error





# Elif ------------------------------------------------------------------------------------------
# The elif keyword is Python's way of saying "if the previous conditions were not true, then try this condition".

# Example

a = 33
b = 33
if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")



'''
Else ---------------------------------------------------------------------------------------------
The else keyword catches anything which isn't caught by the preceding conditions.
'''

a = 200
b = 33
if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")
else:
  print("a is greater than b")



'''
Short Hand If -------------------------------------------------------------------
If you have only one statement to execute, you can put it on the same line as the if statement.
'''

if a > b : print("YES")


'''
Short Hand If
If you have only one statement to execute, you can put it on the same line as the if statement.
'''

print("YEs") if b > a else print("NO")

# This technique is known as Ternary Operators, or Conditional Expressions.

# Example
# One line if else statement, with 3 conditions:

a = 330
b = 330
print("A") if a > b else print("=") if a == b else print("B")


'''
Nested If
You can have if statements inside if statements, this is called nested if statements
'''

x = 41

if x > 10:
  print("Above ten,")
  if x > 20:
    print("and also above 20!")
  else:
    print("but not above 20.")




'''
The pass Statement
if statements cannot be empty, but if you for some reason have an if statement with no content, put in the pass statement to avoid getting an error.

Example: 
'''

a = 33
b = 200

if b > a:
  pass