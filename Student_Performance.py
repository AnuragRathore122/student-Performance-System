import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. LOAD DATA

data = pd.read_csv("student_performance.csv")

print("Dataset Shape:")
print(data.shape)

# 2. BASIC DATA ANALYSIS

print("\nFirst 5 rows:")
print(data.head())

print("\nData Types:")
print(data.dtypes)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Rows:")
print(data.duplicated().sum())

print("\nGrade Distribution:")
print(data["grade"].value_counts())

print("\nStatistical Summary:")
print(data.describe())

# 3. VISUALIZATION

data["grade"].value_counts().plot(kind="bar")

plt.title("Grade Distribution")
plt.xlabel("Grade")
plt.ylabel("Number of Students")

plt.show()

# 4. SELECT FEATURES

X = data[
    [
        "weekly_self_study_hours",
        "attendance_percentage",
        "class_participation",
        "total_score"
    ]
]

# 5. SELECT TARGET

y = data["grade"]

# 6. ENCODE TARGET

encoder = LabelEncoder()

y = encoder.fit_transform(y)

print("\nGrade Classes:")
print(encoder.classes_)

# 7. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining Size:", len(X_train))
print("Testing Size:", len(X_test))

# 8. FEATURE SCALING

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# 9. LOGISTIC REGRESSION

logistic = LogisticRegression(
    max_iter=1000
)

logistic.fit(
    X_train_scaled,
    y_train
)

logistic_pred = logistic.predict(
    X_test_scaled
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

# 10. KNN

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(
    X_train_scaled,
    y_train
)

knn_pred = knn.predict(
    X_test_scaled
)

knn_accuracy = accuracy_score(
    y_test,
    knn_pred
)

# 11. DECISION TREE

tree = DecisionTreeClassifier(
    random_state=42
)

tree.fit(
    X_train,
    y_train
)

tree_pred = tree.predict(
    X_test
)

tree_accuracy = accuracy_score(
    y_test,
    tree_pred
)

# 12. RANDOM FOREST

forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

forest.fit(
    X_train,
    y_train
)

forest_pred = forest.predict(
    X_test
)

forest_accuracy = accuracy_score(
    y_test,
    forest_pred
)

# 13. MODEL COMPARISON

print("\nMODEL ACCURACY")
print("==============================")

print(
    "Logistic Regression:",
    logistic_accuracy
)

print(
    "KNN:",
    knn_accuracy
)

print(
    "Decision Tree:",
    tree_accuracy
)

print(
    "Random Forest:",
    forest_accuracy
)

# 14. CLASSIFICATION REPORT

print("\nRANDOM FOREST REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        forest_pred,
        target_names=encoder.classes_
    )
)

# 15. CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    forest_pred
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")
plt.title("Random Forest Confusion Matrix")

plt.show()

# 16. NEW STUDENT PREDICTION

new_student = np.array([
    [
        20,     # weekly study hours
        90,     # attendance
        6,      # participation
        88      # total score
    ]
])


prediction = forest.predict(
    new_student
)


predicted_grade = encoder.inverse_transform(
    prediction
)


print("\nNEW STUDENT PREDICTION")
print("==============================")

print(
    "Predicted Grade:",
    predicted_grade[0]
)