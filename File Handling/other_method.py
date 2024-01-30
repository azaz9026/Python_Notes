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


## closed method ------------------------------------------------------------------------------------------------

f = open("data.text" , 'r')
print(f.read())
f.close()


## Tell method --------------------------------------------------------------------------------------------------


f = open("data.text" , 'r')
content = f.read()
position = f.tell()
print(f'{content} and it position is that {position}')
f.close()


## Seek method --------------------------------------------------------------------------------------------------


f = open("data.text" , 'r')
f.seek(1)
content = f.read()
position = f.tell()
print(f'{content} and it position is that {position}')
f.close()


with open('data.text' , 'r+') as file:
    file.truncate(1)
    data = file.read()
    print(data)


with open('data.text' , 'r') as source , open('testing.text' , 'w') as destination:
    destination.write(source.read())

with open('testing.text' , 'r') as file:
    cont = file.read()
    print(cont)