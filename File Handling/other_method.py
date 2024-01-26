# other Method --------------------------------------------------------------------------------------------

## closed method

with open("data.text" , 'w') as file:
    file.write('this is a closed Method')
    file.close()


with open("data.text" , 'r') as file:
    data = file.read()
    file.close()

print(data)