import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
ds = load_boston()
#print(ds.keys())
#print(ds.DESCR)
df = pd.DataFrame(ds.data, columns = ds.feature_names)
#print(df.head(3))

#add a column with the target variable MEDV
df['MEDV'] = ds.target
#print(df.head(3))

#check for missing values
df.isnull().sum()

#EDA
sns.distplot(df['MEDV'], bins=30)
plt.show()
#looks like normally distributed with a few  outliers

#measure the linear relationships between variables with a correlation matrix
corr_matrix = df.corr().round(2)
sns.heatmap(data = corr_matrix, annot = True) #annot=True enables annotation inside the squares

#to fit the lin regress model, we select the features with a high correlation to the target variable (here LSTAT, RM)
#check for multicolinearity: features with high correlation with each other should not be used together to train the model (here RAD/TAX, DIS/INDUS, DIS/NOX, DIS/AGE)

plt.figure(figsize=(20, 5))
features = ['LSTAT', 'RM']
target = df['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = df[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title('Variation in house price')
    plt.xlabel(col)
    plt.ylabel('House price (MEDV)')

#concatenate the LSTAT and RM columns using np.c_ 
X = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT','RM'])
y = df['MEDV']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#training and testing the model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

#evaluating the model for training set
from sklearn.metrics import r2_score
y_train_predict = lin_model.predict(X_train)
r2 = r2_score(Y_train, y_train_predict)
print(r2)

#evaluating the model for testing set
y_test_predict = lin_model.predict(X_test)
r2 = r2_score(Y_test, y_test_predict)
print(r2)
