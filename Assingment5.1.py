import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

X_selected = X[['bmi', 's5', 'bp']]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE with bmi, s5, and bp: {mse}")
print(f"R2 Score with bmi, s5, and bp: {r2}")


X_extended = X[['bmi', 's5', 'bp', 'age', 'sex']]
X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X_extended, y, test_size=0.2, random_state=42)

model.fit(X_train_ext, y_train_ext)
y_pred_ext = model.predict(X_test_ext)

mse_ext = mean_squared_error(y_test_ext, y_pred_ext)
r2_ext = r2_score(y_test_ext, y_pred_ext)
print(f"MSE with more variables: {mse_ext}")
print(f"R2 Score with more variables: {r2_ext}")
