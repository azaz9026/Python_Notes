# other Method --------------------------------------------------------------------------------------------

## closed method

with open("data.text" , 'w') as file:
    file.write('this is a closed Method')
    file.close()


with open("data.text" , 'r') as file:
    data = file.read()
    file.close()

print(data)


## tell method ------------------------------------------------------------------------------------------------

with open("data.text" , 'r') as file:
    data = file.read(5)
    position = file.tell()
    print( f' data :- {data} , Position :- {position} ' )


## seek method ------------------------------------------------------------------------------------------------


with open("data.text" , 'r') as file:
    file.seek(3)
    data = file.read(5)
    print(data)


## truncate method --------------------------------------------------------------------------------------------


with open("data.text" , 'r+') as file:
    file.truncate(5)
    data = file.read()
    print(data)


## readable method -----------------------------------------------------------------------------------------------


with open("data.text" , 'r') as file:
    print( f' this file is readable' , file.readable())
   

with open("data.text" , 'r') as file:
    print( f' this file is readable' , file.writable())


with open("data.text" , 'r') as file:
    print( f' this file is readable' , file.seekable())

