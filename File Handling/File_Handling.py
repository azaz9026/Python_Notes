# File Handling ---------------------------------------------------------------------------------------------------------------

'''
File Handling
The key function for working with files in Python is the open() function.

The open() function takes two parameters; filename, and mode.

There are four different methods (modes) for opening a file:
'''

'''
"r" - Read - Default value. Opens a file for reading, error if the file does not exist

"a" - Append - Opens a file for appending, creates the file if it does not exist

"w" - Write - Opens a file for writing, creates the file if it does not exist

"x" - Create - Creates the specified file, returns an error if the file exists
'''

# Syntax
# To open a file for reading it is enough to specify the name of the file:

# f = open("demofile.txt")

# f = open("demofile.txt" , "rt")


## Write Operation ---------------------------

def greeting(name):
    return f"Hello , how are you {name}"


data = greeting("Azaz")
print(data)


with open("data.text" , 'w') as file:
    file.write(data)

with open("data.text" , 'r') as file:
    print(file.read())


# Write a Multiple line of data ---------------------------------------------

line  = ['line1 \n' , 'line2 \n' , 'line3 \n']

with open('text_line.text' , 'w') as file:
    file.writelines(line)

with open('text_line.text' , 'r') as file:
    for lines in file.readlines():
        print(lines)

new_line = "\n this is a new line"

with open('text_line.text' , 'a') as file:
    file.write(new_line)

with open('text_line.text' , 'r') as file:
    for lines in file.readlines():
        print(lines)  


# Reading a File ---------------------------------------------------------------

with open("data.text" , 'r') as file:
    print(file.read(10))
    