## Data Toolkit-------------------------------------------------------------------------------------

## -------------------------------------------------------------------------Practice Questions----------------------------------------------------------------------------------

'''
Que 1 :- Demonstrate three different methods for creating identical 2D arrays in NumPy Provide the code for each 
method and the final output after each method?
'''

# Using np.zeros:

import numpy as np

# Method 1: Using np.zeros
array_method1 = np.zeros((3, 4))

print("Array created using np.zeros:")
print(array_method1)

# Using np.ones:

import numpy as np

# Method 2: Using np.ones
array_method2 = np.ones((3, 4))

print("Array created using np.ones:")
print(array_method2)

# Using np.full:

import numpy as np

# Method 3: Using np.full
array_method3 = np.full((3, 4), fill_value=5)

print("Array created using np.full:")
print(array_method3)




'''
Que 1 :-  Using the Numpy function, generate an array of 100 evenly spaced number Petween 1 and 10 and 
Reshape that 1D array into a 2D array?
'''



import numpy as np

# Generate the evenly spaced array
array_1d = np.linspace(1, 10, 100)

# Reshape the 1D array into a 2D array
array_2d = array_1d.reshape((10, 10))  # Reshape into a 10x10 array

print("1D Array:")
print(array_1d)
print("\n2D Array:")
print(array_2d)




'''
np.array :- np.array is a function in NumPy used to create arrays. It takes a Python list or tuple as input and converts it into a NumPy array.
            The resulting array is typically a new NumPy array with its own memory allocation.

np.asarray :- np.asarray is a function in NumPy that performs a similar task to np.array. However,
              it has an additional parameter dtype that allows you to specify the data type of the array. 
              If the input is already a NumPy array, np.asarray does not create a copy by default, it just returns the input array.

np.asanyarray :- np.asanyarray is another function in NumPy that is similar to np.asarray. 
                 The difference is that np.asanyarray does not always create a new array. If the input is already
                 a NumPy array, it returns it as is. However, if the input is any other array-like object, it converts it into a NumPy array.
'''


'''

Deep Copy :- Deep copy means creating a new object and then recursively copying the content of the original object into the new object.
             In Python, you can achieve deep copy using the copy.deepcopy() function from the copy module. Deep copy creates a completely independent
             copy of the original object, including all nested objects. Any changes made to the copy will not affect the original, and vice versa.

Shallow Copy :- Shallow copy, on the other hand, creates a new object, but instead of recursively copying the content of nested objects, 
                it simply references the nested objects from the original object. In Python, you can achieve shallow copy using the copy.copy() function from the copy module. 
'''



'''
Que 4 :-generate a 3x3 array with random floating-point numbers between 0.5 and 20, and then round each number in the array to 2 decimal places?
'''


import numpy as np

# Generate the random floating-point array
random_array = np.random.uniform(0.5, 20, (3, 3))

# Round each number in the array to 2 decimal places
rounded_array = np.round(random_array, decimals=2)

print("Original Array:")
print(random_array)
print("\nRounded Array:")
print(rounded_array)




## Que 5:--------------------------------------------------------------------------------------------------------------------------------



import numpy as np

# Define the parameters
w = 1  # Start value
wR = 100  # End value
m = 5  # Number of rows
n = 5  # Number of columns

# Create the NumPy array with random integers
random_array = np.random.randint(w, wR, size=(m, n))

print("Original Array:")
print(random_array)

# Extract all even integers from array
even_numbers = random_array[random_array % 2 == 0]

# Extract all odd integers from array
odd_numbers = random_array[random_array % 2 != 0]

print("\nEven Numbers:")
print(even_numbers)

print("\nOdd Numbers:")
print(odd_numbers)


## Que 6 :----------------------------------------------------------------------------------------------------------------------------------


import numpy as np

# Create the 3D NumPy array with random integers between 1 and 10
random_3d_array = np.random.randint(1, 11, size=(3, 3, 3))

print("Original 3D Array:")
print(random_3d_array)

# a) Find the indices of the maximum values along each depth level (third axis)
max_indices = np.argmax(random_3d_array, axis=2)

print("\nIndices of maximum values along each depth level:")
print(max_indices)

# b) Perform element-wise multiplication between both arrays
multiplied_array = random_3d_array * random_3d_array

print("\nElement-wise multiplication of the array with itself:")
print(multiplied_array)



## Que 7 :------------------------------------------------------------------------------------------------------------------------------


import pandas as pd

# Sample dataset
data = {'Name': ['John', 'Alice', 'Bob'],
        'Age': [25, 30, 35],
        'Phone': ['(123) 456-7890', '(987) 654-3210', '(111) 222-3333']}

# Create DataFrame
df = pd.DataFrame(data)

# Clean and transform the 'Phone' column
df['Phone'] = df['Phone'].str.replace(r'\D', '').astype(int)

# Display table attributes
print("Table Attributes:")
print(df.info())




## Que 8 :----------------------------------------------------------------------------------------------------------------------------------


import pandas as pd

# a) Read the 'data.csv' file using pandas, skipping the first 50 rows
df = pd.read_csv('data.csv', skiprows=range(1, 51))

# b) Only read the columns: 'Last Name', 'Gender', 'Email', 'Phone', and 'Salary' from the file
selected_columns = ['Last Name', 'Gender', 'Email', 'Phone', 'Salary']
df = df[selected_columns]

# c) Display the first 10 rows of the filtered dataset
print("First 10 rows of the filtered dataset:")
print(df.head(10))

# d) Extract the ‘Salary’ column as a Series and display its last 5 values
salary_series = df['Salary']
print("\nLast 5 values of the 'Salary' column:")
print(salary_series.tail(5))




## Que 9:---------------------------------------------------------------------------------------------------------------------------------


import pandas as pd

# Load the dataset
df = pd.read_csv('People_Dataset.csv')

# Filter and select rows
filtered_df = df[(df['Last Name'].str.contains('Du\e', case=False)) & 
                 (df['Gender'] == 'Female') &
                 (df['Salary'] < 1000)]  # Assuming you meant less than 1000 for the salary

# Display the filtered DataFrame
print(filtered_df)



## Que 10:--------------------------------------------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np

# Generate a series of 35 random integers between 1 and 6
random_series = pd.Series(np.random.randint(1, 7, size=35))

# Reshape the series into a 7x5 DataFrame
df = pd.DataFrame(random_series.values.reshape(7, 5))

print(df)




## Que 11:---------------------------------------------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np

# Create the first Series with random numbers ranging from 10 to 50
series1 = pd.Series(np.random.randint(10, 51, size=50))

# Create the second Series with random numbers ranging from 100 to 1000
series2 = pd.Series(np.random.randint(100, 1001, size=50))

# Create the DataFrame by joining the two Series by column
df = pd.DataFrame({'col1': series1, 'col2': series2})

# Display the DataFrame
print(df)



## Que 12:---------------------------------------------------------------------------------------------------------------------------------


import pandas as pd

# Load the dataset
df = pd.read_csv('People_Dataset.csv')

# a) Delete the 'Email', 'Phone', and 'Date of birth' columns from the dataset
df.drop(['Email', 'Phone', 'Date of birth'], axis=1, inplace=True)

# b) Delete the rows containing any missing values
df.dropna(inplace=True)

# c) Print the final output
print(df)



## Que 13:---------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Generate random float values for x and y
x = np.random.rand(100)
y = np.random.rand(100)

# Create scatter plot
plt.scatter(x, y, color='red', marker='o', label='Random Points')

# Add horizontal line at y = 0.5
plt.axhline(y=0.5, color='blue', linestyle='--', label='y = 0.5')

# Add vertical line at x = 0.5
plt.axvline(x=0.5, color='green', linestyle=':', label='x = 0.5')

# Label the axes
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Set plot title
plt.title('Advanced Scatter Plot of Random Values')

# Display legend
plt.legend()

# Show plot
plt.show()




## Que 14 :-----------------------------------------------------------------------------------------------------------------



import pandas as pd
import numpy as np

# Create a date range
dates = pd.date_range('2024-01-01', periods=1000)

# Generate random data for temperature and humidity
temperature = np.random.normal(loc=25, scale=5, size=1000)
humidity = np.random.normal(loc=50, scale=10, size=1000)

# Create a DataFrame
df = pd.DataFrame({'Date': dates, 'Temperature': temperature, 'Humidity': humidity})



import matplotlib.pyplot as plt
import seaborn as sns

# Create NumPy array data containing 1000 samples from a normal distribution
data = np.random.normal(size=1000)


# Set the title of the plot
plt.title('Histogram with PDF Overlay')

# Plot the histogram
plt.hist(data, bins=30, density=True, alpha=0.5, color='g')

# Plot the PDF (probability density function) overlay
sns.kdeplot(data, color='b')

plt.show()



# Create two random arrays
x = np.random.randn(100)
y = np.random.randn(100)

# Determine the quadrant of each point
quadrant = np.sign(x) * np.sign(y)

# Scatter plot with colors based on quadrant
plt.scatter(x, y, c=quadrant, cmap='viridis', label=['Q1', 'Q2', 'Q3', 'Q4'])

# Add legend
plt.legend()

# Label the axes
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Set the title
plt.title('Quadrant-wise Scatter Plot')

plt.show()


# Plotting temperature on left y-axis
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Temperature', color=color)
ax1.plot(df['Date'], df['Temperature'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Creating another y-axis for humidity
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Humidity', color=color)
ax2.plot(df['Date'], df['Humidity'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Temperature and Humidity Over Time')
fig.tight_layout()  
plt.show()


## Que 15 :-----------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a NumPy array data containing 1000 samples from a normal distribution
data = np.random.normal(size=1000)

# Plot histogram with PDF overlay
plt.hist(data, bins=30, density=True, alpha=0.5, color='g', label='Histogram')
sns.kdeplot(data, color='b', label='PDF Overlay')
plt.title('Histogram with PDF Overlay')
plt.legend()
plt.show()


## Que 16 :-----------------------------------------------------------------------------------------------------------------

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Create two random arrays
x = np.random.randn(100)
y = np.random.randn(100)

# Determine the quadrant of each point
quadrant = np.sign(x) * np.sign(y)

# Scatter plot with colors based on quadrant
sns.scatterplot(x=x, y=y, hue=quadrant, palette='viridis', legend='full')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Quadrant-wise Scatter Plot')
plt.legend(title='Quadrant')
plt.show()



## Que 17 :----------------------------------------------------------------------------------------------------------------- 


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Create two random arrays
x = np.random.randn(100)
y = np.random.randn(100)

# Determine the quadrant of each point
quadrant = np.sign(x) * np.sign(y)

# Scatter plot with colors based on quadrant
sns.scatterplot(x=x, y=y, hue=quadrant, palette='viridis', legend='full')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Quadrant-wise Scatter Plot')
plt.legend(title='Quadrant')
plt.show()


## Que 18 :----------------------------------------------------------------------------------------------------------------- 


from bokeh.plotting import figure, output_file, show
import numpy as np

# Generate data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Create a new plot with title and axis labels
p = figure(title="Sine Wave Function", x_axis_label='x', y_axis_label='sin(x)', plot_width=600, plot_height=400)

# Add a line renderer with legend and line thickness
p.line(x, y, legend_label="sin(x)", line_width=2)

# Add grid lines
p.grid.grid_line_alpha = 0.3

# Show the result
output_file("sine_wave.html")
show(p)



## Que 19 :----------------------------------------------------------------------------------------------------------------- 


from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource

# Generate random categorical data
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(1, 20, size=len(categories))

# Create a ColumnDataSource
source = ColumnDataSource(data=dict(categories=categories, values=values))

# Create a new plot with title and axis labels
p = figure(x_range=categories, title="Random Categorical Bar Chart", x_axis_label='Categories', y_axis_label='Values', plot_height=400)

# Add bars
p.vbar(x='categories', top='values', width=0.9, source=source, legend_label="values", line_color='white', fill_color='values')

# Add hover tooltips
p.add_tools(HoverTool(tooltips=[("Category", "@categories"), ("Value", "@values")]))

# Show the result
output_file("random_categorical_bar_chart.html")
show(p)



## Que 20 :----------------------------------------------------------------------------------------------------------------- 


import plotly.graph_objects as go
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)
y = np.random.randn(100)

# Create a basic line plot
fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))

# Set axis labels and title
fig.update_layout(xaxis_title='X-axis', yaxis_title='Y-axis', title='Simple Line Plot')

# Show the result
fig.show()



## Que 21 :----------------------------------------------------------------------------------------------------------------- 


import plotly.graph_objects as go
import numpy as np

# Generate random data
labels = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(1, 100, size=len(labels))

# Create a pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', title='Interactive Pie Chart')])

# Show the result
fig.show()
