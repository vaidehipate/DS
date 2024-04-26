'''Group A : Assignment - 8**


### Problem Statement -
* **Data Visualization I**
  1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information about
the passengers who boarded the unfortunate Titanic ship. Use the Seaborn library to see if we
can find any patterns in the data.
 2. Write a code to check how the price of the ticket (column name: 'fare') for each passenger
is distributed by plotting a histogram.'''

# Importing the Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

### Understanding the Dataset - Titanic
'''The Titanic dataset is a classic dataset in data science
with around 891 passengers and features like passenger class,
age, sex, and most importantly, survival information. It's used for
exploring data analysis techniques, building models to predict survival,
and understanding factors that affected survival rates.'''

# Loading the Dataset

df = pd.read_csv("/content/titanic_dataset.csv")

# Understanding the data
df.head()

# Getting the columns names
df.columns

# Getting the overall information of the data
df.info()

# From the above information we can observe that **Age, Cabin, Embarked** columns contains missing values

### What are we trying to solve ?
# This dataset is most commonly used to predict whether the person survived or not based on the other input attributes.
# So the main task is to find the relation between the **Survived** Column i.e Target Column and the other input columns.



### Question 1
''' Use the inbuilt dataset 'titanic'.
The dataset contains 891 rows and contains information
about the passengers who boarded the unfortunate Titanic ship.
Use the Seaborn library to see if we can find any patterns in the data.'''

#1. Categorical Columns Vs Target Column**

#Using a **Bar Chart** to see if the Survival Rate differed based on the Categorical columns i.e. Pclass, Sex, etc.


# Pclass
df['Pclass'].value_counts()

# Pclass contains 3 unique classes

sns.barplot(x='Pclass',y='Survived',data=df)

# Based on the above information, we can observe that the survival rate of the 1st Class passengers is more, compared to the other classes.

# Sex
df['Sex'].value_counts()

sns.barplot(x='Sex',y='Survived', data=df)

#The Survival Rate of Females is more as compared to the Males.

#2. Continuous Columns Vs Survival Rate**

# Using a **KDE Plot** to identify the relation between continuous columns i.e. Age, etc and the Target Column.

# AGE
df['Age'].value_counts()

sns.kdeplot(data=df[df['Survived']==1], x='Age', label="Survived")
sns.kdeplot(data=df[df['Survived']==0], x='Age', label="Not Survived")

# From the above graph, one can say that it is difficult to infer the Survival of the passenger based on their Age.

# PairPlot -> To explore the Relationships between all pairs of numerical features in the dataset

sns.pairplot(df)


###Question 2
# Write a code to check how the price of the ticket (column name: 'fare') for each passenger is distributed by plotting a histogram.

# Getting the count of unqiue values of the Fare column
df['Fare'].value_counts()

# Creating a Histogram of the Fare Prices
sns.histplot(df['Fare'],bins='auto')

# Adding Title and Labels
plt.xlabel("Fare Price")
plt.ylabel("Number of Passengers")
plt.title("Distribution of Fare Prices on the Titanic")
plt.show()

