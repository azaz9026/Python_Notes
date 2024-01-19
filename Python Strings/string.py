# Python Strings -------------------------------------------------------------------------------------------------------------------------------

'''
Strings in python are surrounded by either single quotation marks, or double quotation marks.

'hello' is the same as "hello".

You can display a string literal with the print() function:
'''

print("Hello")
print('Hello')


# Assign String to a Variable ---------------------------------------------------------------------------------------------------------------------

a = "Md azaz"
print(a)

# Multiline Strings-----------------------------------------------------------------------------------------------------------------------------------
# double Or three single quotes:

a = """Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua."""
print(a)


'''
Strings are Arrays ------------------------------------------------------------------------------------------------------------------------------------
Like many other popular programming languages, strings in Python are arrays of bytes representing unicode characters.

However, Python does not have a character data type, a single character is simply a string with a length of 1.

Square brackets can be used to access elements of the string.
'''

# Get the character at position 1 (remember that the first character has the position 0):

a = "Hello, World!"
print(a[0])



# Looping Through a String-----------------------------------------------------------------------------------------------------------------------------
# Since strings are arrays, we can loop through the characters in a string, with a for loop.

for i in "Md Azaz":
    print(i)



# String Length --------------------------------------------------------------------------------------------------------------------------------------
# To get the length of a string, use the len() function.

a = "Hello, World!"
print(len(a))


# Check String ----------------------------------------------------------------------------------------------------------------------------------------
# To check if a certain phrase or character is present in a string, we can use the keyword in.

txt = "The best things in life are free!"
print("free" in txt)

# Print only if "free" is present:

txt = "The best things in life are free!"
if "free" in txt:
  print("Yes, 'free' is present.")


# Check if NOT -----------------------------------------------------------------------------------------------------------------------------------------
# To check if a certain phrase or character is NOT present in a string, we can use the keyword not in.

txt = "The best things in life are free!"
print("azaz" not in txt)

txt = "The best things in life are free!"
if "azaz" not in txt:
  print("No, 'expensive' is NOT present.")
