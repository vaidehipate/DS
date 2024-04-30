'''3) Descriptive Statistics - Measures of Central Tendency and variability
Perform the following operations on any open source dataset (e.g., data.csv)
1. Provide summary statistics (mean, median, minimum, maximum, standard deviation) for a
dataset (age, income etc.) with numeric variables grouped by one of the qualitative
(categorical) variable. For example, if your categorical variable is age groups and quantitative
variable is income, then provide summary statistics of income grouped by the age groups.
Create a list that contains a numeric value for each response to the categorical variable.
2. Write a Python program to display some basic statistical details like percentile, mean,
standard deviation etc. of the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’ of
iris.csv dataset.
'''

import pandas as pd
import numpy as np
from statistics import mean, median, mode

#df = pd.read_csv("https://github.com/mwaskom/seaborn-data/raw/master/iris.csv")
df = pd.read_csv("/content/iris.csv")

df.head()

#Mean

Mean =np.mean(df)
print(f"Mean :{Mean}")

#if this didnt worked out , use this . above code will work out on jupyter
'''
# Replace non-numeric values with NaN
iris_numeric = df.apply(pd.to_numeric, errors='coerce')

# Calculate the mean of each numerical column
mean_values = iris_numeric.mean()

print("Mean values:")
print(mean_values)
'''


# Median
median_value = np.median(df['sepal_length'])
print(f"Median: {median_value}")

#mode
'''
The mode() function calculates the mode, which is the value that appears most frequently in a dataset.
Unlike mean and median, mode may not always exist or may not be unique.
If there is a single mode (i.e., one value appears most frequently), mode() returns that value.
If there are multiple modes (i.e., multiple values with the same highest frequency), mode() raises a StatisticsError.
    '''

try:
    mode_value = mode(df['sepal_length'])
    print(f"Mode: {mode_value}")
except StatisticsError:
    print("No unique mode")

#max

maxm = np.max(df['sepal_length'])
print(maxm)

#min

minm = np.min(df['sepal_length'])
print(minm)

print(df.mode())

# Mid-Range
mid_range = (max(df['sepal_length']) + min(df['sepal_length'])) / 2
print(f"Mid-Range: {mid_range}")

#groupby
df.groupby(['sepal_length']).mean()

#if this didnt worked out , use this . above code will work out on jupyter
'''
df_numeric = df[pd.to_numeric(df['sepal_length'], errors='coerce').notnull()]

# Convert 'sepal_length' column to numeric
df_numeric['sepal_length'] = pd.to_numeric(df_numeric['sepal_length'])

# Calculate the mean of 'sepal_length'
mean_sepal_length = df_numeric['sepal_length'].mean()
'''

df.groupby(['sepal_length']).count()


#Quantiles:

# calculate the quartiles (25th, 50th and 75th percentiles)
quartiles = df['sepal_length'].quantile([0.25, 0.5, 0.75])
print("Quartiles:", quartiles)
print("\n")

# calculate the deciles (10th, 20th, ..., 90th percentiles)
deciles = df['sepal_length'].quantile(np.arange(0.1, 1, 0.1))
print("Deciles:", deciles)
print("\n")


# calculate the percentiles (1st, 2nd, ..., 99th percentiles)
percentiles = df['sepal_length'].quantile(np.arange(0.01, 1, 0.01))
print("Percentiles:", percentiles)
print("\n")


#using builtin function

df.std()
df.var()

##if this didnt worked out , use this . above code will work out on jupyter
'''
# Assuming df is your DataFrame
# Convert all columns to numeric (errors='coerce' will convert non-numeric values to NaN)
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Calculate variance and standard deviation
variance = df_numeric.var()
std_deviation = df_numeric.std()

print("Variance:")
print(variance)
print("\nStandard Deviation:")
print(std_deviation)
'''

# Calculating mean without using libraries
math_s = df['sepal_length']
def my_mean(math_s):
    return sum(math_s) / len(math_s)
print("mean : ",my_mean(math_s))

# Calculating median without using libraries
def my_median(math_s):
    n = len(math_s)
    index = n // 2
    # Sample with an odd number of observations
    if n % 2:
        return sorted(math_s)[index]
    # Sample with an even number of observations
    return sum(sorted(math_s)[index - 1:index + 1]) / 2
print("median : ",my_median(math_s))

# Calculating mode without using libraries
from collections import Counter
def my_mode(math_s):
    c = Counter(math_s)
    return [k for k, v in c.items() if v == c.most_common(1)[0][1]]
print("Mode : ",my_mode(math_s))

'''Explaination:
c.items(): This part of the code retrieves a list of (key, value) pairs from the Counter object c. Each key represents an element in the input list math_s, and its corresponding value represents the count of occurrences of that element.
for k, v in c.items(): This part of the code iterates over each (key, value) pair in the list obtained from c.items(). Here, k represents the element (key), and v represents the count (value).
if v == c.most_common(1)[0][1]: This part of the code checks if the count v is equal to the count of the most common element in the list.
c.most_common(1) retrieves a list of the most common element(s) and their counts from the Counter object c.
[0] accesses the first element in this list.
[1] accesses the count associated with that element.
'''


# Calculating Standard deviation and variation without using libraries

import math

#calculating variance
def variance(math_s):
  n = len(math_s)
  my_mean = sum(math_s) / n
  deviations = [(x - my_mean) ** 2 for x in math_s]
  variance = sum(deviations) / n
  return variance

#calculating standard deviation
def stdev(math_s):
  var = variance(math_s)
  std_dev = math.sqrt(var)
  return std_dev

print("Standard Deviation : %s"% (stdev(math_s)))

