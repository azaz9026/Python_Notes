

'''
                                                                                 ( Statistics ) 
============================================================================= (Practice Questions) =========================================================================
'''

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










''' 
Let's start by loading the tips dataset from the seaborn library and then move on to calculating the skewness, covariance, Pearson correlation coefficient, and visualizing the data.
'''
# 1. Load the Tips Dataset

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






'''
Problem 1: A/B Testing for Website Layout (Z-Test)
Setup:
We want to check if the new website layout leads to a higher conversion rate using the data from the old and new layouts.
'''


import numpy as np
from statsmodels.stats.weightstats import ztest

# Generate data for old and new layouts
old_layout = np.array([1] * 50 + [0] * 950)  # 50 purchases out of 1000 visitors
new_layout = np.array([1] * 70 + [0] * 930)  # 70 purchases out of 1000 visitors

# Perform two-sample z-test
z_stat, p_value = ztest(new_layout, old_layout, value=0, alternative='larger')

print("Z-statistic:", z_stat)
print("P-value:", p_value)

# Conclusion
if p_value < 0.05:
    print("The new layout has a significantly higher conversion rate.")
else:
    print("There is no significant difference between the old and new layouts.")






'''
Problem 2: Tutoring Service Exam Scores (Z-Test)
Setup:
We want to determine if the tutoring program significantly improves students' exam scores by comparing the scores before and after the program.
'''

import numpy as np
from statsmodels.stats.weightstats import ztest

# Generate data for before and after the program
before_program = np.array([75, 80, 85, 70, 90, 78, 92, 88, 82, 87])
after_program = np.array([80, 85, 90, 80, 92, 80, 95, 90, 85, 88])

# Perform paired z-test (one-sample z-test on the difference)
score_diff = after_program - before_program
z_stat, p_value = ztest(score_diff, value=0, alternative='larger')

print("Z-statistic:", z_stat)
print("P-value:", p_value)

# Conclusion
if p_value < 0.05:
    print("The tutoring service significantly improves exam scores.")
else:
    print("There is no significant improvement in exam scores.")







'''
Problem 3: Blood Pressure Reduction by Drug (Z-Test)
Setup:
We want to check if the new drug significantly reduces blood pressure by comparing the measurements before and after the drug administration.
'''

import numpy as np
from statsmodels.stats.weightstats import ztest

# Generate data for before and after drug administration
before_drug = np.array([145, 150, 140, 135, 155, 160, 152, 148, 130, 138])
after_drug = np.array([130, 140, 132, 128, 145, 148, 138, 136, 125, 130])

# Perform paired z-test (one-sample z-test on the difference)
bp_diff = before_drug - after_drug
z_stat, p_value = ztest(bp_diff, value=0, alternative='larger')

print("Z-statistic:", z_stat)
print("P-value:", p_value)

# Conclusion
if p_value < 0.05:
    print("The drug significantly reduces blood pressure.")
else:
    print("The drug does not significantly reduce blood pressure.")






'''
Problem 1: Z-Test for Customer Service Response Time
The customer service department claims that the average response time is less than 5 minutes. We will use a one-sample z-test to verify this claim.
'''

import numpy as np
from statsmodels.stats.weightstats import ztest

# Generate the response times
response_times = np.array([4.3, 3.8, 5.1, 4.9, 4.7, 4.2, 5.2, 4.5, 4.6, 4.4])

# Perform one-sample z-test (null hypothesis: mean >= 5)
z_stat, p_value = ztest(response_times, value=5, alternative='smaller')

print("Z-statistic:", z_stat)
print("P-value:", p_value)

# Conclusion
if p_value < 0.05:
    print("The claim that the average response time is less than 5 minutes is supported.")
else:
    print("The claim that the average response time is less than 5 minutes is not supported.")






'''
Problem 2: A/B Test for Website Layouts (T-Test)
We will perform an independent two-sample t-test to determine if one layout has a higher click-through rate.
'''

import scipy.stats as stats

# Data for layout A and layout B
layout_a_clicks = [28, 32, 33, 29, 31, 34, 30, 35, 36, 37]
layout_b_clicks = [40, 41, 38, 42, 39, 44, 43, 41, 45, 47]

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(layout_a_clicks, layout_b_clicks)

# Degrees of freedom
df = len(layout_a_clicks) + len(layout_b_clicks) - 2

print("T-statistic:", t_stat)
print("P-value:", p_value)
print("Degrees of Freedom:", df)

# Conclusion
if p_value < 0.05:
    print("There is a significant difference in click-through rates between the two layouts.")
else:
    print("There is no significant difference in click-through rates between the two layouts.")





'''
Problem 3: T-Test for Drug Effectiveness
We will conduct an independent two-sample t-test to compare the cholesterol levels between the new drug and the existing drug.
'''

import scipy.stats as stats

# Cholesterol levels for existing and new drugs
existing_drug_levels = [180, 182, 175, 185, 178, 176, 172, 184, 179, 183]
new_drug_levels = [170, 172, 165, 168, 175, 173, 170, 178, 172, 176]

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(existing_drug_levels, new_drug_levels)

print("T-statistic:", t_stat)
print("P-value:", p_value)

# Conclusion
if p_value < 0.05:
    print("The new drug is significantly more effective than the existing drug.")
else:
    print("There is no significant difference between the new and existing drugs.")














'''
Problem 4: T-Test for Educational Intervention
We will use a paired t-test to compare the pre- and post-intervention math scores.
'''


import scipy.stats as stats

# Pre- and post-intervention test scores
pre_intervention_scores = [80, 85, 90, 75, 88, 82, 92, 78, 85, 87]
post_intervention_scores = [90, 92, 88, 92, 95, 91, 96, 93, 89, 93]

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(pre_intervention_scores, post_intervention_scores)

print("T-statistic:", t_stat)
print("P-value:", p_value)

# Conclusion
if p_value < 0.05:
    print("The intervention significantly improved the test scores.")
else:
    print("The intervention did not significantly improve the test scores.")










'''
Problem 1: T-Test for Gender-Based Salary Gap
We will compare the salaries of male and female employees using an independent two-sample t-test to determine if there is a statistically significant difference in their average salaries.
'''

import numpy as np
import scipy.stats as stats

# Generate synthetic salary data for male and female employees
np.random.seed(0)
male_salaries = np.random.normal(loc=50000, scale=10000, size=20)
female_salaries = np.random.normal(loc=55000, scale=9000, size=20)

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(male_salaries, female_salaries)

print("T-statistic:", t_stat)
print("P-value:", p_value)

# Conclusion
if p_value < 0.05:
    print("There is a statistically significant difference in average salaries between male and female employees.")
else:
    print("There is no statistically significant difference in average salaries between male and female employees.")









'''
Problem 2: T-Test for Product Quality Scores
We will perform an independent two-sample t-test to compare the quality scores of two product versions.
'''

# Data for quality scores of two product versions
version1_scores = [85, 88, 82, 89, 87, 84, 90, 88, 85, 86, 91, 83, 87, 84, 89, 86, 84, 88, 85, 86, 89, 90, 87, 88, 85]
version2_scores = [80, 78, 83, 81, 79, 82, 76, 80, 78, 81, 77, 82, 80, 79, 82, 79, 80, 81, 79, 82, 79, 78, 80, 81, 82]

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(version1_scores, version2_scores)

print("T-statistic:", t_stat)
print("P-value:", p_value)

# Conclusion
if p_value < 0.05:
    print("There is a statistically significant difference in the quality scores between the two product versions.")
else:
    print("There is no statistically significant difference in the quality scores between the two product versions.")









'''
Problem 3: T-Test for Customer Satisfaction Scores
We will use an independent two-sample t-test to compare customer satisfaction scores between two branches of a restaurant chain.
'''

# Data for customer satisfaction scores for two branches
branch_a_scores = [4, 5, 3, 4, 5, 4, 5, 3, 4, 4, 5, 4, 4, 3, 4, 5, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 3, 4, 5, 4]
branch_b_scores = [3, 4, 2, 3, 4, 3, 4, 2, 3, 3, 4, 3, 3, 2, 3, 4, 4, 3, 2, 3, 4, 3, 2, 4, 3, 3, 4, 2, 3, 4, 3]

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(branch_a_scores, branch_b_scores)

print("T-statistic:", t_stat)
print("P-value:", p_value)

# Conclusion
if p_value < 0.05:
    print("There is a statistically significant difference in customer satisfaction between the two branches.")
else:
    print("There is no statistically significant difference in customer satisfaction between the two branches.")






''' 
Problem 4: Chi-Square Test for Age Groups and Voter Preferences
We will perform a Chi-Square test to determine if there is a significant association between age groups and voter preferences.
'''

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Generate synthetic data for age groups and voter preferences
np.random.seed(0)
age_groups = np.random.choice(['18-30', '31-50', '51+'], size=30)
voter_preferences = np.random.choice(['Candidate A', 'Candidate B'], size=30)

# Create a contingency table
data = pd.DataFrame({'Age Group': age_groups, 'Voter Preference': voter_preferences})
contingency_table = pd.crosstab(data['Age Group'], data['Voter Preference'])

# Perform Chi-Square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print("Chi-Square statistic:", chi2)
print("P-value:", p_value)

# Conclusion
if p_value < 0.05:
    print("There is a significant association between age groups and voter preferences.")
else:
    print("There is no significant association between age groups and voter preferences.")




'''
22. Chi-Square Test for Product Satisfaction Levels vs. Customer Regions
We will perform a Chi-Square test to determine if there is a significant relationship between product satisfaction levels and customer regions using the given contingency table.
'''


import numpy as np
from scipy.stats import chi2_contingency

# Sample data: Product satisfaction levels (rows) vs. Customer regions (columns)
data = np.array([[50, 30, 40, 20], [30, 40, 30, 50], [20, 30, 40, 30]])

# Perform Chi-Square test
chi2_stat, p_value, dof, expected = chi2_contingency(data)

print("Chi-Square statistic:", chi2_stat)
print("P-value:", p_value)
print("Degrees of Freedom:", dof)
print("Expected Frequencies Table:")
print(expected)

# Conclusion
if p_value < 0.05:
    print("There is a significant relationship between product satisfaction levels and customer regions.")
else:
    print("There is no significant relationship between product satisfaction levels and customer regions.")




'''
23. Chi-Square Test for Job Performance Levels Before and After Training
We will perform a Chi-Square test to determine if there is a significant difference in job performance levels before and after the training using the given contingency table.
'''

# Sample data: Job performance levels before (rows) and after (columns) training
data = np.array([[50, 30, 20], [30, 40, 30], [20, 30, 40]])

# Perform Chi-Square test
chi2_stat, p_value, dof, expected = chi2_contingency(data)

print("Chi-Square statistic:", chi2_stat)
print("P-value:", p_value)
print("Degrees of Freedom:", dof)
print("Expected Frequencies Table:")
print(expected)

# Conclusion
if p_value < 0.05:
    print("There is a significant difference in job performance levels before and after training.")
else:
    print("There is no significant difference in job performance levels before and after training.")




'''  
24. ANOVA Test for Customer Satisfaction Scores Among Product Versions
We will perform an ANOVA test to determine if there is a significant difference in customer satisfaction scores among three different product versions.
'''


from scipy.stats import f_oneway

# Sample data: Customer satisfaction scores for each product version
standard_scores = [80, 85, 90, 78, 88, 82, 92, 78, 85, 87]
premium_scores = [90, 92, 88, 92, 95, 91, 96, 93, 89, 93]
deluxe_scores = [95, 98, 92, 97, 96, 94, 98, 97, 92, 99]

# Perform ANOVA test
f_stat, p_value = f_oneway(standard_scores, premium_scores, deluxe_scores)

print("F-statistic:", f_stat)
print("P-value:", p_value)

# Conclusion
if p_value < 0.05:
    print("There is a significant difference in customer satisfaction scores among the three product versions.")
else:
    print("There is no significant difference in customer satisfaction scores among the three product versions.")
