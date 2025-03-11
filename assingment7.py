import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Problem 1: Support Vector Machine (SVM)

df_svm = pd.read_csv("data_banknote_authentication.csv")


X_svm = df_svm.drop(columns=["class"])
y_svm = df_svm["class"]


X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y_svm, test_size=0.2, random_state=20)


svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_svm, y_train_svm)
y_pred_linear = svm_linear.predict(X_test_svm)

print("SVM with Linear Kernel:")
print(confusion_matrix(y_test_svm, y_pred_linear))
print(classification_report(y_test_svm, y_pred_linear))


svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train_svm, y_train_svm)
y_pred_rbf = svm_rbf.predict(X_test_svm)

print("SVM with RBF Kernel:")
print(confusion_matrix(y_test_svm, y_pred_rbf))
print(classification_report(y_test_svm, y_pred_rbf))

# Problem 2: Decision Tree

df_dt = pd.read_csv("suv.csv")

X_dt = df_dt[["Age", "EstimatedSalary"]]
y_dt = df_dt["Purchased"]

# Split data
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_dt, y_dt, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_dt = scaler.fit_transform(X_train_dt)
X_test_dt = scaler.transform(X_test_dt)

dt_entropy = DecisionTreeClassifier(criterion='entropy')
dt_entropy.fit(X_train_dt, y_train_dt)
y_pred_entropy = dt_entropy.predict(X_test_dt)

print("Decision Tree with Entropy Criterion:")
print(confusion_matrix(y_test_dt, y_pred_entropy))
print(classification_report(y_test_dt, y_pred_entropy))

dt_gini = DecisionTreeClassifier(criterion='gini')
dt_gini.fit(X_train_dt, y_train_dt)
y_pred_gini = dt_gini.predict(X_test_dt)

print("Decision Tree with Gini Criterion:")
print(confusion_matrix(y_test_dt, y_pred_gini))
print(classification_report(y_test_dt, y_pred_gini))
