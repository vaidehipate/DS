'''10) Data Visualization III
Download the Iris flower dataset or any other dataset into a DataFrame. (e.g.,
https://archive.ics.uci.edu/ml/datasets/Iris ). Scan the dataset and give the inference as:
1. List down the features and their types (e.g., numeric, nominal) available in the dataset.
2. Create a histogram for each feature in the dataset to illustrate the feature distributions.
3. Create a boxplot for each feature in the dataset.
4. Compare distributions and identify outliers.
'''

# Importing the Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import warnings
warnings.filterwarnings('ignore')

### Understanding the Dataset - Iris
* The Iris dataset consists of four independent features representing various measurements related to the morphology of iris flowers, namely sepal length, sepal width, petal length, and petal width. The dataset is structured as a classification problem, where the goal is to predict the class or species of an iris flower based on these four feature values.

# Loading the Dataset

iris = load_iris()

df = pd.DataFrame(iris['data'],columns=iris['feature_names'])

# Understanding the data
df.head()

# Getting the columns names
df.columns

# Getting the overall information of the data
df.info()

df.describe(

)

* From the above information we can observe that their are no Missing Values.

### What are we trying to solve ?
* Here, we are trying to get the basic Overview of the Iris Dataset. Knowing the Types of features that are present, identifying outliers and comparing data distributions


<hr>

### Question 1
* List down the features and their types (e.g., numeric, nominal) available in the dataset.

df.dtypes

* Iris Dataset consist of four features each of Float DataType.

### Question 2
* Create a histogram for each feature in the dataset to illustrate the feature distributions.

# Sepal Length
df['sepal length (cm)'].hist()
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.title("Data Distribution of Sepal Length Feature")

# Sepal Width
df['sepal width (cm)'].hist()
plt.xlabel("Sepal Width")
plt.ylabel("Frequency")
plt.title("Data Distribution of Sepal Width Feature")

# Petal Length
df['petal length (cm)'].hist()
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.title("Data Distribution of Petal Length Feature")

# Petal Width
df['petal width (cm)'].hist()
plt.xlabel("Petal Width")
plt.ylabel("Frequency")
plt.title("Data Distribution of Petal Width Feature")

<hr>

###Question 3

  * Create a boxplot for each feature in the dataset.

df.plot(kind="box", subplots=True, layout=(2, 3), figsize=(12, 6))
plt.suptitle("Boxplots of Iris Flower Features")
plt.show()

### Question 4

*  Compare distributions and identify outliers.

### Distributions -
* The distribution of Sepal Width Feature follows a **Binomial distribution**.
* Petal Length Feature exhibits a **Bimodal distribution**.
* Sepal Length Feature displays **Right Skewness**.

### Outliers -
* On observing the above Plots, we can infer that only **Sepal Width** Columns contains Outliers.

