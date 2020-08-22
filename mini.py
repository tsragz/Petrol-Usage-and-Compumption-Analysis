#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#dataset operations
dataset = pd.read_csv('petrol.csv') #amend path depending on location
dataset.head()
dataset.describe()
dataset.shape()

#plotting
dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

#data preperation
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#intercept
print(regressor.intercept_)

#slope
print(regressor.coef_)

#predictions
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Act': y_test, 'Pred': y_pred})
df

#Final evaluation of metrics
from sklearn import metrics
print('Mean Absolute Err:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Err:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Err:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

