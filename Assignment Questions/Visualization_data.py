# MATPLOTLIB ASSIGNMENT: ------------------------------------------------------------------------------------------------------------------------------

''' (Use Matplotlib for the visualization of the given questions) '''



import matplotlib.pyplot as plt

# Task 1: Scatter plot
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 5, 7, 6, 8, 9, 10, 12, 13]
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.show()

# Task 2: Line plot
plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Plot')
plt.show()

# Task 3: Bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 40, 30, 35, 20]
plt.bar(categories, values)
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.title('Bar Chart')
plt.show()

# Task 4: Histogram
import numpy as np
data = np.random.normal(loc=0, scale=1, size=1000)  # Generate random data
plt.hist(data, bins=30)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# Task 5: Pie chart
sections = ['Section A', 'Section B', 'Section C', 'Section D']
sizes = [25, 30, 15, 30]
plt.pie(sizes, labels=sections, autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()




# SEABORN ASSIGNMENT: -------------------------------------------------------------------------------------------------------------------------------

''' (Use Seaborn for the visualization of the given questions) '''





# Additional Task 1: Scatter plot with synthetic dataset
synthetic_x = np.random.rand(50)
synthetic_y = 2 * synthetic_x + np.random.rand(50)
plt.scatter(synthetic_x, synthetic_y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Synthetic Data')
plt.show()

# Additional Task 2: Distribution of random numbers
random_data = np.random.normal(loc=0, scale=1, size=1000)
plt.hist(random_data, bins=30)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Random Numbers')
plt.show()

# Additional Task 3: Comparison of categories based on values
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(1, 100, size=len(categories))
plt.bar(categories, values)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Comparison of Categories Based on Values')
plt.show()

# Additional Task 4: Distribution of numerical variable across categories
categories = ['A', 'B', 'C', 'D', 'E']
numeric_values = np.random.normal(loc=0, scale=1, size=len(categories))
plt.bar(categories, numeric_values)
plt.xlabel('Categories')
plt.ylabel('Numerical Values')
plt.title('Distribution of Numerical Variable Across Categories')
plt.show()

# Additional Task 5: Correlation matrix heatmap
import seaborn as sns

correlated_features = np.random.randn(10, 5)
correlation_matrix = np.corrcoef(correlated_features, rowvar=False)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()



# PLOTLY ASSIGNMENT: -------------------------------------------------------------------------------------------------------------------------------

''' (Use Plotly for the visualization of the given questions) '''




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np

# Task 1: 3D Scatter Plot
np.random.seed(30)
data = {
    'X': np.random.uniform(-10, 10, 300),
    'Y': np.random.uniform(-10, 10, 300),
    'Z': np.random.uniform(-10, 10, 300)
}
df = pd.DataFrame(data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['X'], df['Y'], df['Z'])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')
plt.show()

# Task 2: Violin Plot
np.random.seed(15)
data = {
    'Grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 200),
    'Score': np.random.randint(50, 100, 200)
}
df = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
sns.violinplot(x='Grade', y='Score', data=df)
plt.xlabel('Grade')
plt.ylabel('Score')
plt.title('Violin Plot of Student Grades')
plt.show()

# Task 3: Heatmap
np.random.seed(20)
data = {
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100),
    'Day': np.random.choice(range(1, 31), 100),
    'Sales': np.random.randint(1000, 5000, 100)
}
df = pd.DataFrame(data)

sales_pivot = df.pivot_table(index='Month', columns='Day', values='Sales', aggfunc='sum')
plt.figure(figsize=(10, 6))
sns.heatmap(sales_pivot, cmap='YlGnBu')
plt.title('Sales Variation Across Months and Days')
plt.xlabel('Day')
plt.ylabel('Month')
plt.show()



# BOKEH ASSIGNMENT: ---------------------------------------------------------------------------------------------------------------------

'''(Use Bokeh for the visualization of the given questions)''' 




from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource
import numpy as np
import pandas as pd

# Task 1: Bokeh plot displaying a sine wave
output_notebook()

x = np.linspace(0, 10, 100)
y = np.sin(x)

p = figure(title="Sine Wave", x_axis_label='X', y_axis_label='sin(X)', width=600, height=400)
p.line(x, y, legend_label='sin(x)', line_width=2)
show(p)

# Task 2: Bokeh scatter plot using randomly generated x and y values
np.random.seed(0)
n = 100
x = np.random.rand(n) * 10
y = np.random.rand(n) * 10
sizes = np.random.rand(n) * 20
colors = np.random.choice(["red", "blue", "green"], n)

source = ColumnDataSource(data={'x': x, 'y': y, 'sizes': sizes, 'colors': colors})

p = figure(title="Random Scatter Plot", x_axis_label='X', y_axis_label='Y', width=600, height=400)
p.scatter('x', 'y', source=source, size='sizes', color='colors', alpha=0.6)
show(p)

# Task 3: Bokeh bar chart representing the counts of different fruits
fruits = ['Apples', 'Oranges', 'Bananas', 'Pears']
counts = [20, 25, 30, 35]

p = figure(x_range=fruits, plot_height=350, title="Fruit Counts")
p.vbar(x=fruits, top=counts, width=0.9)
p.xgrid.grid_line_color = None
p.y_range.start = 0
show(p)

# Task 4: Bokeh histogram to visualize the distribution of the given data
data_hist = np.random.randn(1000)
hist, edges = np.histogram(data_hist, bins=30)

p = figure(title="Histogram of Random Data", width=600, height=400)
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.5)
show(p)

# Task 5: Bokeh heatmap using the provided dataset
data_heatmap = np.random.rand(10, 10)
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
xx, yy = np.meshgrid(x, y)

p = figure(title="Heatmap", x_range=(0, 1), y_range=(0, 1), width=600, height=400)
p.image(image=[data_heatmap], x=0, y=0, dw=1, dh=1, palette="Viridis256")
show(p)
