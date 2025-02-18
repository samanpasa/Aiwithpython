import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

startups = pd.read_csv('50_Startups.csv', delimiter=',')

print(startups.head())
print(startups.info())

numeric_data = startups.select_dtypes(include=[np.number])

sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.show()

X = startups[['R&D Spend', 'Marketing Spend', 'Administration']]
y = startups['Profit']

for col in X.columns:
    plt.scatter(X[col], y)
    plt.xlabel(col)
    plt.ylabel('Profit')
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training RMSE: {np.sqrt(train_mse)}")
print(f"Testing RMSE: {np.sqrt(test_mse)}")
print(f"Training R2 Score: {train_r2}")
print(f"Testing R2 Score: {test_r2}")
