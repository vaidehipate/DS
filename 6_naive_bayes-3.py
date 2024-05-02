'''6) Data Analytics III
1. Implement Simple Naïve Bayes classification algorithm using Python/R on iris.csv dataset.
2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall on
the given dataset.'''




### Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')

### Loading Dataset

from seaborn import load_dataset

df = load_dataset('iris')

df

### Understanding The Dataset

df.shape

df.columns

df.info()

df.describe()

### Intialization

#Initializing Independent And Dependent Variables as X and y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X

y


### Data Preprocessing And Splitting into Testing and Training samples

#Using Min-Max Scalar to Scale the X variable
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()
X_scaled = scalar.fit_transform(X)
X_scaled

#Splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

## ----------------------------------------------------------------------------------------------------------------------------------

### Naive Bayes Classification

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

y_pred

y_test


### Confusion Matrix And Classification Report

from sklearn.metrics import confusion_matrix,classification_report

labels = ['setosa','versicolor','virginica']
cm = confusion_matrix(y_test,y_pred,labels = labels)
cm

#Visualization of Confusion matrix using Seaborn
plt.figure(figsize=(15,8))
sns.heatmap(cm,annot=True,
            xticklabels=['Setosa','Versicolor','Virginica'],
            yticklabels=['Setosa','Versicolor','Virginica'],)
plt.ylabel('Prediction')
plt.xlabel('Actual')
plt.title("Confusion Matrix")
plt.show()

report = classification_report(y_test,y_pred)
print(f"Classification report: \n\n {report}")


### Calculating True Positive,True Negative,False Positive,False Negative values for each Class

#For Setosa Class

tp = cm[0, 0]
fn = cm[0, 1] + cm[0, 2]
fp = cm[1, 0] + cm[2, 0]
tn = cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2]

print(f"Outcome Values:\nTrue Positive : {tp} \nFalse Negative : {fn} \nFalse Positive : {fp} \nTrue Negative : {tn}")

#For Versicolor Class

tp = cm[1, 1]
fn = cm[1, 0] + cm[1, 2]
fp = cm[0, 1] + cm[2, 1]
tn = cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2]

print(f"Outcome Values:\nTrue Positive : {tp} \nFalse Negative : {fn} \nFalse Positive : {fp} \nTrue Negative : {tn}")

#For Virginica Class

tp = cm[2, 2]
fn = cm[2, 0] + cm[2, 1]
fp = cm[0, 2] + cm[1, 2]
tn = cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1]

print(f"Outcome Values:\nTrue Positive : {tp} \nFalse Negative : {fn} \nFalse Positive : {fp} \nTrue Negative : {tn}")

### Calculating Accuracy,Precision,Recall,F1-Score

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

print("Accuracy : ",accuracy_score(y_test,y_pred))
print("Precision : ",precision_score(y_test,y_pred,labels = labels,pos_label=1,average = 'micro'))
print("Recall : ",recall_score(y_test,y_pred,labels = labels,pos_label=1,average = 'micro'))
print("F1-Score : ",f1_score(y_test,y_pred,labels = labels,pos_label=1,average = 'micro'))
