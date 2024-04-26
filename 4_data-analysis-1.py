'''4) Data Analytics I
Create a Linear Regression Model using Python/R to predict home prices using Boston Housing
Dataset (https://www.kaggle.com/c/boston-housing). The Boston Housing dataset contains
information about various houses in Boston through different parameters. There are 506 samples and
14 feature variables in this dataset.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


#df = pd.read_csv("https://github.com/selva86/datasets/raw/master/BostonHousing.csv")
df = pd.read_csv("/content/BostonHousing.csv")


df.head()

df.info(

)
df.describe()

#crim columns does not relevant for our analysis.

df.drop('crim', axis = 1, inplace=True)

sns.pairplot(df, vars = ['rm', 'zn', 'dis', 'chas','medv'])




df.plot.scatter('rad','tax')

plt.subplots(figsize=(12,8))
sns.heatmap(df.corr(), cmap = 'RdGy')

sns.pairplot(df, vars = ['lstat', 'ptratio', 'indus', 'tax', 'nox', 'rad', 'age', 'medv'])



Trainning Linear Regression Model¶
Define X and Y

X: Varibles named as predictors, independent variables, features.
Y: Variable named as response or dependent variable

X = df[[ 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio',  'lstat']]
y = df['medv']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


sns.distplot((y_test-predictions),bins=50);

coefficients = pd.DataFrame(lm.coef_,X.columns)
coefficients.columns = ['coefficients']
coefficients
