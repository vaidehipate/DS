''' **Group A : Assignment - 9**


### Problem Statement -
Data Visualization II
  1. Use the inbuilt dataset 'titanic' as used in the above problem. Plot a box plot for distribution of
age with respect to each gender along with the information about whether they survived or
not. (Column names : 'sex' and 'age')
  2. Write observations on the inference from the above statistics.
  '''

# Importing the Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

### Understanding the Dataset - Titanic
#The Titanic dataset is a classic dataset in data science with around 891 passengers and features
#like passenger class, age, sex, and most importantly, survival information. It's used for exploring
#data analysis techniques, building models to predict survival, and understanding factors that affected survival rates.

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
'''Here, we are trying to get the Data Distribution of the Age wrt Gender Column 
and to infer if their is any connection of the Distribution with the Target Column.'''




### Question 1
'''Use the inbuilt dataset 'titanic' as used in the above problem.
Plot a box plot for distribution of age with respect to each gender along with 
the information about whether they survived or not. (Column names : 'sex' and 'age')'''

# Knowing the Age Column
df['Age'].value_counts()

# Getting the Data Distribution of the Age column
df['Age'].hist()

# Knowing the Gender Column
df['Sex'].value_counts()

# Checking for the Outliers
df['Sex'].isna().sum()

# Creating the Required BoxPlot
sns.boxplot(
    x = 'Sex',
    y = 'Age',
    hue = 'Survived',
    showmeans = True,
    data = df
)

plt.title("Distribution of Age by Sex and Survived on the Titanic")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.show()



###Question 2

  #Write observations on the inference from the above statistics.

### **Inferences**


#Age Distribution
  
  # In general, passengers on the Titanic tended to be on the younger side, with a median age around 30.
  # There seems to be a wider range of ages for males compared to females. This means that male passengers had a larger spread in their ages, with some very young and some very old, while females had a more concentrated age group.

##Survival and Age**
  # These observations show potential relationships, but they don't necessarily mean age caused survival rates
  # In females, the spread of ages is larger among survivors compared to non-survivors. This means there was more variation in the ages of people who survived the disaster.


