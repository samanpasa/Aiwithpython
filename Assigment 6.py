import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Read the CSV file
df = pd.read_csv("bank.csv", delimiter=';')

# Step 2: Select relevant columns
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]

# Step 3: Convert categorical variables to dummy variables
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'], drop_first=True)

# Convert target variable 'y' to binary (yes=1, no=0)
df3['y'] = df3['y'].map({'yes': 1, 'no': 0})

# Step 4: Produce a heatmap of correlation coefficients
plt.figure(figsize=(10, 6))
sns.heatmap(df3.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Step 5: Define target variable and explanatory variables
X = df3.drop(columns=['y'])
y = df3['y']

# Step 6: Split dataset into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 7: Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Step 8: Confusion Matrix and Accuracy Score for Logistic Regression
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("Logistic Regression Accuracy Score:", accuracy_score(y_test, y_pred_log))

# Step 9: K-Nearest Neighbors Model (k=3)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Confusion Matrix and Accuracy Score for KNN
print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("KNN Accuracy Score:", accuracy_score(y_test, y_pred_knn))

# Step 10: Compare Results
"""
- Logistic Regression typically performs well when features are linearly separable.
- KNN can perform better or worse depending on the value of k.
- Compare accuracy scores and confusion matrices to determine which model is better.
"""
