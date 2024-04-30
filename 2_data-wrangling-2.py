'''2) Data Wrangling II
Create an “Academic performance” dataset of students and perform the following operations using
Python.
1. Scan all variables for missing values and inconsistencies. If there are missing values and/or
inconsistencies, use any of the suitable techniques to deal with them.
2. Scan all numeric variables for outliers. If there are outliers, use any of the suitable techniques
to deal with them.
3. Apply data transformations on at least one of the variables. The purpose of this
transformation should be one of the following reasons: to change the scale for better
understanding of the variable, to convert a non-linear relation into a linear one, or to decrease
the skewness and convert the distribution into a normal distribution.'''




import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn




import warnings 
warnings.filterwarnings('ignore')

#using random creating dataset values

np.random.seed(42)

data = {
    'Student_ID': range(1, 21),
    'Name': [f'Student_{i}' for i in range(1, 21)],
    'Age': np.random.randint(18, 25, size=20),
    'Gender': np.random.choice(['Male', 'Female'], size=20),
    'Math_Score': np.random.randint(50, 100, size=20),
    'English_Score': np.random.randint(40, 95, size=20),
    'Physics_Score': np.random.randint(30, 85, size=20),
    'Absenteeism': np.random.choice([0, 1], size=20),
    'Examgiven':np.random.choice([0,1],size=20)
}

df=pd.DataFrame(data)

#Adding null values inside the dataset

df.loc[df['Absenteeism'] == 1, 'English_Score'] = np.nan
df.loc[df['Absenteeism'] == 1, 'Physics_Score'] = np.nan
df.loc[df['Absenteeism'] == 1, 'Math_Score'] = np.nan

print(df)



#filling / replacing null values with mean , median

df['Math_Score'].fillna(df['Math_Score'].mean(), inplace=True)
df['English_Score'].fillna(df['English_Score'].median(), inplace=True)
df['Physics_Score'].fillna(df['Physics_Score'].median(), inplace=True)
print(df)

# showing outliers using boxplot for columns 1.) using seaborn

sns.boxplot(df['Age'])

sns.boxplot(df['English_Score'])

# showing outliers using boxplot for columns 2.) using pandas


df.boxplot("Physics_Score")

df.boxplot(column=["Physics_Score", "Math_Score"])

newdf = df[df["Physics_Score"]>55.0]
newdf.boxplot("Physics_Score")

df.loc[df['Absenteeism'] == 1, 'Physics_Score'] = np.nan
print(df)

df['Physics_Score'].fillna(df['Physics_Score'].median(), inplace=True)
print(df)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df['Age'], df['English_Score'])

# x-axis label
ax.set_xlabel('(Age)')

# y-axis label
ax.set_ylabel('(English scores )')
plt.show()

sns.boxplot(df['Math_Score'])


sns.boxplot(df['Physics_Score'])


# Calculate the upper and lower limits
Q1 = df['Math_Score'].quantile(0.25)
Q3 = df['Math_Score'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

# Create arrays of Boolean values indicating the outlier rows
upper_array = np.where(df['Math_Score'] >= upper)[0]
lower_array = np.where(df['Math_Score'] <= lower)[0]

# Removing the outliers
df.drop(index=upper_array, inplace=True)
df.drop(index=lower_array, inplace=True)

# Print the new shape of the DataFrame
print("New Shape: ", df.shape)

sns.boxplot(df['Physics_Score'])

print(df)



67.833333**0.5

from scipy.stats import skew

# Calculate skewness before transformation
original_skewness = skew(df["Math_Score"])

# Apply transformation (e.g., Square Root Transformation)
transformed_data = np.sqrt(df["Math_Score"])

print(original_skewness)

#Calculate skewness after transformation
transformed_skewness = skew(transformed_data)

print("Original Skewness:", original_skewness)
print("Transformed Skewness:", transformed_skewness)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox, skew, norm
import seaborn as sns

# Generating a skewed dataset
np.random.seed(42)
skewed_data = np.random.exponential(size=1000)

skewed_data

# Apply Box-Cox transformation
transformed_data, lambda_value = boxcox(skewed_data + 1)  # Adding 1 to handle zero values if present



# Calculate skewness before transformation
skewness_before = skew(skewed_data)

skewness_before



# Calculate skewness after transformation
skewness_after = skew(transformed_data)

skewness_after

# Plotting original and transformed distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(skewed_data, kde=True)
plt.title(f'Skewed Distribution (Skewness: {skewness_before:.2f})')



plt.subplot(1, 2, 2)
sns.histplot(transformed_data, kde=True)
plt.title(f'Transformed Normal Distribution (Skewness: {skewness_after:.2f})')


plt.tight_layout()
plt.show()
