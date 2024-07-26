# Python Classes and Objects -------------------------------------------------------------------------------------------------------------------------

'''
Python Classes/Objects ------------------------------------------------------------
Python is an object oriented programming language.

Almost everything in Python is an object, with its properties and methods.

A Class is like an object constructor, or a "blueprint" for creating objects.
'''

'''
Create a Class------------
To create a class, use the keyword class:
'''

class myclass:
    x = 4
'''
Create Object-------------
Now we can use the class named MyClass to create objects:
'''

p1 = myclass()
print(p1.x)


# The __init__() Function -------------------------------------------------------------
'''
Example
Create a class named Person, use the __init__() function to assign values for name and age:
'''

class person :
    def __init__(self , name , age):
        self.name = name
        self.age = age


p1 = person("Md Azaz" , 21)

print(p1.name)
print(p1.age)

print(f"my name is {p1.name} and I am {p1.age} year old")

# Note: The __init__() function is called automatically every time the class is being used to create a new object.


'''
The __str__() Function
The __str__() function controls what should be returned when the class object is represented as a string.

If the __str__() function is not set, the string representation of the object is returned:
'''

class person :
    def __init__(self , name , age):
        self.name = name
        self.age = age

# p2 = person("Lux" , 21)

# print(p2)

# The string representation of an object WITH the __str__() function:

class person :
    def __init__(self , name  , age):
        self.name = name
        self.age  = age

    def __str__(self):
        return f"my is {self.name} and I am {self.age} year old ....."
    
p2 = person("Lux" , 21)
print(p2)


'''
Object Methods ------------------------------------
Objects can also contain methods. Methods in objects are functions that belong to the object.

Let us create a method in the Person class:
'''

class person :
    def __init__(self , name  , age):
        self.name = name
        self.age  = age

    def myfun(self):
        print("Hello my name is " + self.name)

p3 = person("OM" , 20)
p3.myfun()


'''
The self Parameter ----------------------------------
The self parameter is a reference to the current instance of the class, and is used to access variables that belongs to the class.

It does not have to be named self , you can call it whatever you like, but it has to be the first parameter of any function in the class:
'''

class Person:
  def __init__(mysillyobject, name, age):
    mysillyobject.name = name
    mysillyobject.age = age

  def myfunc(abc):
    print("Hello my name is " + abc.name)

p1 = Person("John", 36)
p1.myfunc()


'''
Modify Object Properties ---------------------------------
You can modify properties on objects like this:
'''
p1.age = 50

print(p1.age)



'''
Delete Objects -------------------------------------------
You can delete objects by using the del keyword:
'''

del p1
# print(p1)


'''
The pass Statement ---------------------------------------
class definitions cannot be empty, but if you for some reason have a class definition with no content, put in the pass statement to avoid getting an error.
'''

class new_person:
    pass
