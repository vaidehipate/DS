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

### **CONFUSION MATRIX**

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

print(f"Confusion Matrix: \n {confusion_matrix(y_test,y_pred)}")
print(f"Accuracy Score: {accuracy_score(y_test,y_pred)}")

# Plot the confusion matrix using heatmap
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,xticklabels=['Not Purchased','Purchased'],yticklabels=['Not Purchased','Purchased'])
plt.ylabel('Prediction')
plt.xlabel('Actual')
plt.title("Confusion Matrix")
plt.show()

# Classification Report for metrics of confusion matrix
print(f"Confusion Matrix Metrics: \n {classification_report(y_test,y_pred)}")



