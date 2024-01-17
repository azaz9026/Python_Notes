# Python Iterators -----------------------------------------------------------------------------------------------------------------------------

'''
Python Iterators ---------------------------------------------------

An iterator is an object that contains a countable number of values.

An iterator is an object that can be iterated upon, meaning that you can traverse through all the values.

Technically, in Python, an iterator is an object which implements the iterator protocol, which consist of the methods __iter__() and __next__().
'''


'''
Iterator vs Iterable
Lists, tuples, dictionaries, and sets are all iterable objects. They are iterable containers which you can get an iterator from.
'''

mytuple = ("apple", "banana", "cherry")
print(dir(mytuple))

my_it = iter(mytuple)

print(next(my_it))
print(next(my_it))
print(next(my_it))
# print(next(my_it)) # StopIteration


mystr = "banana"
myit = iter(mystr)

print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))


# Looping Through an Iterator -----------------------------------------

'''
Example
Iterate the values of a tuple:
'''

mytuple = ("apple", "banana", "cherry")

for x in mytuple:
  print(x)


'''
Example
Iterate the characters of a string:
'''

mystr = "banana"

for x in mystr:
  print(x)


# Create an Iterator ------------------------------------------------------------------------------------------------------------------

class MyNumbers:
  def __iter__(self):
    self.a = 1
    return self

  def __next__(self):
    x = self.a
    self.a += 1
    return x

myclass = MyNumbers()
myiter = iter(myclass)

print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))


'''
StopIteration
The example above would continue forever if you had enough next() statements, or if it was used in a for loop.

To prevent the iteration from going on forever, we can use the StopIteration statement.
'''

class MyNumbers:
  def __iter__(self):
    self.a = 1
    return self

  def __next__(self):
    if self.a <= 20:
      x = self.a
      self.a += 1
      return x
    else:
      raise StopIteration

myclass = MyNumbers()
myiter = iter(myclass)

for x in myiter:
  print(x)