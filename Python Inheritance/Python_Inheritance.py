# Python Inheritance ------------------------------------------------------------------------------------------------------------------

'''
Python Inheritance
Inheritance allows us to define a class that inherits all the methods and properties from another class.

Parent class is the class being inherited from, also called base class.

Child class is the class that inherits from another class, also called derived class.
'''

'''
Create a Parent Class -----------------------
Any class can be a parent class, so the syntax is the same as creating any other class:
'''

class person :
    def __init__(self , fname , lname):
        self.fname = fname
        self.lname = lname

    def printname(self):
        print(self.fname , self.lname)

#Use the Person class to create an object, and then execute the printname method: 

x = person("Md" , "Azaz")
x.printname()


'''
Create a Child Class -------------------------
To create a class that inherits the functionality from another class, send the parent class as a parameter when creating the child class:
'''


class Student(person):
    def __init__(self , fname , lname):
        person.__init__(self , fname , lname)


'''
Use the super() Function ---------------------------
Python also has a super() function that will make the child class inherit all the methods and properties from its parent:
'''

class Person:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

  def printname(self):
    print(self.firstname, self.lastname)

class Student(Person):
  def __init__(self, fname, lname):
    super().__init__(fname, lname)

x = Student("Mike", "Olsen")
x.printname()



'''
Add Properties -----------------------------------

Example
Add a property called graduationyear to the Student class:
'''

class Student(person):
    def __init__(self , fname , lname):
        super().__init__(fname , lname)
        self.graduationyear = 2019
    

x = Student("Lux" , 21)
print(x.graduationyear)


# -------------------------------------------------------------------------------------------------------------------


class Student(Person):
  def __init__(self, fname, lname, year):
    super().__init__(fname, lname)
    self.graduationyear = year

x = Student("Mike", "Olsen", 2019)

# -----------------------------------------------------------------------------------------------------------------------

class Person:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

  def printname(self):
    print(self.firstname, self.lastname)

class Student(Person):
  def __init__(self, fname, lname, year):
    super().__init__(fname, lname)
    self.graduationyear = year

  def welcome(self):
    print("Welcome", self.firstname, self.lastname, "to the class of", self.graduationyear)

x = Student("Mike", "Olsen", 2019)
x.welcome()
