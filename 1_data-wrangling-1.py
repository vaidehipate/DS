'''
1) Data Wrangling, I
Perform the following operations using Python on any open source dataset (e.g., data.csv)
1. Import all the required Python Libraries.
2. Locate open source data from the web (e.g., https://www.kaggle.com). Provide a clear
description of the data and its source (i.e., URL of the web site).
3. Load the Dataset into pandas dataframe.
4. Data Preprocessing: check for missing values in the data using pandas isnull(), describe()
function to get some initial statistics. Provide variable descriptions. Types of variables etc.
Check the dimensions of the data frame.
5. Data Formatting and Data Normalization: Summarize the types of variables by checking the
data types (i.e., character, numeric, integer, factor, and logical) of the variables in the data set.
If variables are not in the correct data type, apply proper type conversions.
6. Turn categorical variables into quantitative variables in Python
'''
1.
#importing libraries

import pandas as pd
import numpy as np

2.
#importing dataset

data = pd.read_csv("/content/AutoData.csv")

3.lOAD dataset 
#displays first 5 rows

data.head()

#displays last 5 rows

data.tail(5)


4. DATA PREPROCESSING
#display info about data colmns

data.info()

#variable description

data.describe()


#finding missing values

data.isnull()


#missing values count

data.isnull().sum()

#check the value is not null

data.notnull()

#count the not null values

data.notnull().sum()

5.
Data Formatting and Data Normalization

#check the datatype of columns

data.dtypes



#check dimensions of dataset

data.shape

#check total elements in dataset

data.size

#check datatype of columns

data.dtypes

#Applying type conversion

data = data.astype({"make":'category',"curbweight":'float64'})
data.info()

#displays first 20 rows

data[:20]

Turning categorical variables into quantitative variables

# categorical variables into quantitative variables

#method1
data['doornumber'].replace(['two', 'four'],
                        [2, 4], inplace=True)

data[:20]

#method2

# Define numerical values for each category in 'doornumber'
doornumber_values = {
    'Two': 2,
    'Three': 3,
    'Four': 4,
    # Add more mappings as needed
}

# Assign numerical values to the 'doornumber' column
data['doornumber_encoded'] = data['doornumber'].map(doornumber_values)

# Print the first few rows of the DataFrame with the encoded column
print(data.head())

















