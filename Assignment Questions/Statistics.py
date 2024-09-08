

''' Part 1: Generate the Integer List (int_list) and Perform Statistical Analysis
Step 1: Generate a list of 100 random integers between 90 and 130 '''


import random

# Generate list of 100 integers between 90 and 130
int_list = [random.randint(90, 130) for _ in range(100)]


# Function to calculate the mean
def calculate_mean(numbers):
    return sum(numbers) / len(numbers)

mean_value = calculate_mean(int_list)



# Function to calculate the median
def calculate_median(numbers):
    numbers = sorted(numbers)
    n = len(numbers)
    if n % 2 == 0:
        return (numbers[n//2 - 1] + numbers[n//2]) / 2
    else:
        return numbers[n//2]

median_value = calculate_median(int_list)



from collections import Counter

# Function to compute the mode
def calculate_mode(numbers):
    counts = Counter(numbers)
    max_count = max(counts.values())
    mode = [k for k, v in counts.items() if v == max_count]
    return mode

mode_value = calculate_mode(int_list)



# Function to calculate weighted mean
def calculate_weighted_mean(values, weights):
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / sum(weights)

# Example weights (randomly assigned)
weights = [random.randint(1, 10) for _ in range(len(int_list))]
weighted_mean_value = calculate_weighted_mean(int_list, weights)



import math

# Function to calculate geometric mean
def calculate_geometric_mean(numbers):
    product = 1
    for num in numbers:
        product *= num
    return product ** (1 / len(numbers))

geometric_mean_value = calculate_geometric_mean(int_list)



# Function to calculate harmonic mean
def calculate_harmonic_mean(numbers):
    return len(numbers) / sum(1 / num for num in numbers)

harmonic_mean_value = calculate_harmonic_mean(int_list)



# Function to calculate midrange
def calculate_midrange(numbers):
    return (min(numbers) + max(numbers)) / 2

midrange_value = calculate_midrange(int_list)



# Function to calculate trimmed mean by excluding a certain percentage of outliers
def calculate_trimmed_mean(numbers, trim_percentage):
    numbers = sorted(numbers)
    trim_count = int(len(numbers) * trim_percentage)
    trimmed_numbers = numbers[trim_count:-trim_count]
    return calculate_mean(trimmed_numbers)

trimmed_mean_value = calculate_trimmed_mean(int_list, 0.05)



''' Part 2: Generate Another List (int_list2) and Perform Advanced Statistical Analysis
Step 1: Generate a list of 500 integers between 200 and 300 '''


int_list2 = [random.randint(200, 300) for _ in range(500)]


# Step 2: Visualization (Frequency & Gaussian distribution, KDE plot)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Plot Frequency & Gaussian distribution
plt.figure(figsize=(10,6))
sns.histplot(int_list2, kde=False, color='blue', bins=20, label='Frequency')
x = np.linspace(min(int_list2), max(int_list2), 500)
plt.plot(x, norm.pdf(x, np.mean(int_list2), np.std(int_list2)), color='red', label='Gaussian Distribution')
plt.legend()
plt.show()

# Plot Frequency with KDE
plt.figure(figsize=(10,6))
sns.histplot(int_list2, kde=True, color='green', bins=20)
plt.show()

# KDE plot & Gaussian distribution
plt.figure(figsize=(10,6))
sns.kdeplot(int_list2, color='green', label='KDE Plot')
plt.plot(x, norm.pdf(x, np.mean(int_list2), np.std(int_list2)), color='red', label='Gaussian Distribution')
plt.legend()
plt.show()


# Variance and standard deviation calculation
def calculate_variance(numbers):
    mean = calculate_mean(numbers)
    return sum((x - mean) ** 2 for x in numbers) / len(numbers)

def calculate_std_dev(numbers):
    return math.sqrt(calculate_variance(numbers))

variance_value = calculate_variance(int_list2)
std_dev_value = calculate_std_dev(int_list2)



# Function to calculate IQR
def calculate_iqr(numbers):
    numbers = sorted(numbers)
    q1 = np.percentile(numbers, 25)
    q3 = np.percentile(numbers, 75)
    return q3 - q1

iqr_value = calculate_iqr(int_list2)


# Function to calculate coefficient of variation
def calculate_cv(numbers):
    return calculate_std_dev(numbers) / calculate_mean(numbers)

cv_value = calculate_cv(int_list2)


# Function to calculate MAD
def calculate_mad(numbers):
    mean = calculate_mean(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)

mad_value = calculate_mad(int_list2)


# Function to calculate Quartile Deviation
def calculate_quartile_deviation(numbers):
    q1 = np.percentile(numbers, 25)
    q3 = np.percentile(numbers, 75)
    return (q3 - q1) / 2

quartile_deviation_value = calculate_quartile_deviation(int_list2)


# Function to calculate range-based coefficient of dispersion
def calculate_dispersion_coefficient(numbers):
    return calculate_range(numbers) / (max(numbers) + min(numbers))

dispersion_coefficient_value = calculate_dispersion_coefficient(int_list2)


'''Summary of Results
The functions above will give you the ability to:

Generate random integer lists.
Calculate various statistics like mean, median, mode, variance, standard deviation, and interquartile range.
Visualize distributions using seaborn and matplotlib.'''


''' 1. Python Class for a Discrete Random Variable
This class calculates the expected value and variance of a discrete random variable based on its probability mass function (PMF).'''


class DiscreteRandomVariable:
    def __init__(self, outcomes, probabilities):
        self.outcomes = outcomes
        self.probabilities = probabilities
    
    def expected_value(self):
        return sum(o * p for o, p in zip(self.outcomes, self.probabilities))
    
    def variance(self):
        mean = self.expected_value()
        return sum((o - mean) ** 2 * p for o, p in zip(self.outcomes, self.probabilities))

# Example
outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6
rv = DiscreteRandomVariable(outcomes, probabilities)
print("Expected Value:", rv.expected_value())
print("Variance:", rv.variance())


# 2. Simulate Rolling Two Six-Sided Dice

import random

def simulate_dice_rolls(n):
    rolls = [random.randint(1, 6) + random.randint(1, 6) for _ in range(n)]
    mean = sum(rolls) / len(rolls)
    variance = sum((x - mean) ** 2 for x in rolls) / len(rolls)
    return mean, variance

n = 10000
mean, variance = simulate_dice_rolls(n)
print("Expected Value (Mean):", mean)
print("Variance:", variance)


# 3. Generate Random Samples from a Probability Distribution (Binomial or Poisson)

import numpy as np

# Function to generate samples from binomial or poisson distribution
def generate_samples(distribution='binomial', n=1000, **kwargs):
    if distribution == 'binomial':
        samples = np.random.binomial(kwargs['n'], kwargs['p'], n)
    elif distribution == 'poisson':
        samples = np.random.poisson(kwargs['lam'], n)
    else:
        raise ValueError("Unsupported distribution")
    
    mean = np.mean(samples)
    variance = np.var(samples)
    return samples, mean, variance

# Example with Binomial
samples, mean, variance = generate_samples(distribution='binomial', n=1000, p=0.5, n_trials=10)
print("Mean (Binomial):", mean)
print("Variance (Binomial):", variance)

# Example with Poisson
samples, mean, variance = generate_samples(distribution='poisson', n=1000, lam=4)
print("Mean (Poisson):", mean)
print("Variance (Poisson):", variance)


# 4. Generate Random Numbers from a Gaussian Distribution


import numpy as np

def gaussian_statistics(n, mean=0, std_dev=1):
    samples = np.random.normal(mean, std_dev, n)
    calculated_mean = np.mean(samples)
    variance = np.var(samples)
    std_dev = np.std(samples)
    return calculated_mean, variance, std_dev

n = 1000
mean, variance, std_dev = gaussian_statistics(n, mean=0, std_dev=1)
print("Mean:", mean)
print("Variance:", variance)
print("Standard Deviation:", std_dev)


# 5. Probability Density Function (PDF) for a Normal Distribution

import math

def normal_pdf(x, mean=0, std_dev=1):
    return (1 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mean) / std_dev) ** 2)

# Example
x_value = 1.5
mean = 0
std_dev = 1
pdf_value = normal_pdf(x_value, mean, std_dev)
print(f"PDF of {x_value} for N({mean}, {std_dev}):", pdf_value)



# 7. Probability Mass Function (PMF) of Poisson Distribution

import math

def poisson_pmf(k, lam):
    return (lam ** k * math.exp(-lam)) / math.factorial(k)

# Example
k_value = 3
lam = 2.0
pmf_value = poisson_pmf(k_value, lam)
print(f"PMF of Poisson distribution at k = {k_value} with λ = {lam}: {pmf_value}")


''' Let's start by loading the tips dataset from the seaborn library and then move on to calculating the skewness, covariance, Pearson correlation coefficient, and visualizing the data.

1. Load the Tips Dataset'''

import seaborn as sns
import pandas as pd

# Load tips dataset from seaborn
tips = sns.load_dataset('tips')

# View the first few rows of the dataset
print(tips.head())


from scipy.stats import skew

# Function to calculate skewness
def calculate_skewness(column):
    return skew(column)

total_bill_skewness = calculate_skewness(tips['total_bill'])
tip_skewness = calculate_skewness(tips['tip'])

print("Skewness of total_bill:", total_bill_skewness)
print("Skewness of tip:", tip_skewness)



# Function to determine the type of skewness
def check_skewness(column_name, skewness_value):
    if skewness_value > 0:
        print(f"{column_name} has positive skewness.")
    elif skewness_value < 0:
        print(f"{column_name} has negative skewness.")
    else:
        print(f"{column_name} is approximately symmetric.")

check_skewness("total_bill", total_bill_skewness)
check_skewness("tip", tip_skewness)



# Function to calculate covariance between two columns
def calculate_covariance(col1, col2):
    return pd.DataFrame({'col1': col1, 'col2': col2}).cov().iloc[0, 1]

covariance = calculate_covariance(tips['total_bill'], tips['tip'])
print("Covariance between total_bill and tip:", covariance)


# Function to calculate Pearson correlation coefficient
def calculate_pearson_corr(col1, col2):
    return pd.DataFrame({'col1': col1, 'col2': col2}).corr().iloc[0, 1]

pearson_corr = calculate_pearson_corr(tips['total_bill'], tips['tip'])
print("Pearson correlation coefficient between total_bill and tip:", pearson_corr)



import matplotlib.pyplot as plt

# Scatter plot for correlation between total_bill and tip
plt.figure(figsize=(8, 6))
plt.scatter(tips['total_bill'], tips['tip'], color='blue', alpha=0.6)
plt.title('Scatter Plot between Total Bill and Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.grid(True)
plt.show()


'''
Summary:
Skewness Calculation: You now have a function to calculate the skewness of total_bill and tip columns.
Skewness Type Check: The program determines whether the columns exhibit positive, negative, or no skewness.
Covariance Calculation: The covariance between total_bill and tip is computed.
Pearson Correlation: The Pearson correlation coefficient between total_bill and tip is calculated.
Scatter Plot Visualization: The correlation between total_bill and tip is visualized using a scatter plot.
'''


