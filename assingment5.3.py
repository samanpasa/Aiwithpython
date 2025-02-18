import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

auto = pd.read_csv('Auto.csv')

X = auto.drop(columns=['mpg', 'name', 'origin'])
y = auto['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = np.logspace(-2, 2, 50)
ridge_scores = []
lasso_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_scores.append(r2_score(y_test, ridge.predict(X_test)))

    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_scores.append(r2_score(y_test, lasso.predict(X_test)))

plt.plot(alphas, ridge_scores, label='Ridge R2')
plt.plot(alphas, lasso_scores, label='Lasso R2')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R2 Score')
plt.legend()
plt.show()

best_alpha_ridge = alphas[np.argmax(ridge_scores)]
best_alpha_lasso = alphas[np.argmax(lasso_scores)]
print(f"Best Ridge Alpha: {best_alpha_ridge}")
print(f"Best Lasso Alpha: {best_alpha_lasso}")