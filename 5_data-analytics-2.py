'''5) Data Analytics II
1. Implement logistic regression using Python/R to perform classification on
Social_Network_Ads.csv dataset.
2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall
on the given dataset.'''


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns



#ads = pd.read_csv('https://github.com/shivang98/Social-Network-ads-Boost/raw/master/Social_Network_Ads.csv')
ads = pd.read_csv('/content/Social_Network_Ads (1).csv')

ads.head()



ads.info()

ads.describe()

ads.isnull().sum()

ads.shape

# Correlation
ads.corr()


# Select only numeric columns
numeric_columns = ads.select_dtypes(include=['int64', 'float64'])

# Calculate correlation
correlation = numeric_columns.corr()

print("Correlation:")
print(correlation)


# Check correlation between dependent and independent variable
plt.figure(figsize=(10,7))  #10-width, 7-height
sns.heatmap(data=ads.corr(),annot=True,center=True,cbar=True)
plt.plot()
# The brighter the color, the stronger the correlation. The darker the color, the weaker the correlation

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
ads1 = ads
ads1.head(10)

ads1['Gender'] = label_encoder.fit_transform(ads1['Gender'])
ads1['Gender'].unique()

'''
 LabelEncoder is used to convert categorical variables into numerical labels.
ads1['Gender'] = label_encoder.fit_transform(ads1['Gender']): This line applies label encoding to the 'Gender' column of the DataFrame ads1.
It replaces categorical values (e.g., 'Male' and 'Female') with numerical labels (e.g., 0 and 1).
'''


np.where(ads1['Purchased'] == 1)

plt.figure(figsize=(10,7))
sns.heatmap(data=ads1.corr(),annot=True,center=True,cbar=True)
plt.plot()

X = ads.iloc[:,[2,3]].values #column 2 & 3 are selected as input variables
y = ads.iloc[:,4].values  #column 4 is selected as target



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to training dataset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)



# Predict the test results
y_pred = classifier.predict(X_test)
y_pred

# **CONFUSION MATRIX**
'''
Confusion Matrix: Provides a tabular representation of the classification results
showing the counts of true positive, true negative, false positive, and false negative predictions.
True Positives (TP): The number of correctly predicted positive instances.
False Positives (FP): The number of incorrectly predicted positive instances.
True Negatives (TN): The number of correctly predicted negative instances.
False Negatives (FN): The number of incorrectly predicted negative instances.
Accuracy: The proportion of correctly classified instances among all instances.
Error Rate: The proportion of incorrectly classified instances among all instances.
Precision: The proportion of true positive predictions among all positive predictions.
Recall: The proportion of true positive predictions among all actual positive instances.
'''

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Extract TP, FP, TN, FN from confusion matrix
TN, FP, FN, TP = cm.ravel()

'''
The confusion matrix cm typically looks like this:

[[TN, FP],
 [FN, TP]]
After applying ravel(), the array is flattened into a one-dimensional array:

[TN, FP, FN, TP]
'''
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)
print("True Positives (TP):", TP)

# Compute Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Compute Error Rate
error_rate = 1 - accuracy
print("Error Rate:", error_rate)

# Compute Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Compute Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)
