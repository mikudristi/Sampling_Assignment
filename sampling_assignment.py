import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
file_path = "Creditcard_data.csv"  # Replace with the actual path
data = pd.read_csv(file_path)

# Explore the dataset
print("Dataset Head:\n", data.head())
print("\nClass Distribution:\n", data['Class'].value_counts())

from imblearn.over_sampling import SMOTE

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the new class distribution
print("\nBalanced Class Distribution:\n", pd.Series(y_resampled).value_counts())

import math

# Parameters
Z = 1.96  # 95% confidence level
p = 0.5   # Proportion of the population
E = 0.05  # Margin of error

# Calculate sample size
n = (Z**2 * p * (1 - p)) / (E**2)
n = math.ceil(n)  # Round up to the nearest whole number
print("\nSample Size:", n)

# Simple Random Sampling
simple_random_sample = data.sample(n=n, random_state=42)
print("\nSimple Random Sample:\n", simple_random_sample.head())

# Systematic Sampling
k = len(data) // n  # Interval
systematic_sample = data.iloc[::k]
print("\nSystematic Sample:\n", systematic_sample.head())

# Stratified Sampling
stratified_sample, _ = train_test_split(data, test_size=(1 - n / len(data)), stratify=data['Class'], random_state=42)
print("\nStratified Sample:\n", stratified_sample['Class'].value_counts())

data_balanced = data.copy()  # Create a copy of the original data
data_balanced['Cluster'] = pd.qcut(data_balanced['V1'], q=10, labels=False)  # Divide into 10 clusters based on 'V1'
selected_clusters = data_balanced['Cluster'].sample(n=2, random_state=42).unique()  # Select 2 random clusters
cluster_sample = data_balanced[data_balanced['Cluster'].isin(selected_clusters)]

# Check class distribution in the cluster sample
class_distribution = cluster_sample['Class'].value_counts()
print("\nCluster Sample Class Distribution:\n", class_distribution)

# If the sample has only one class, resample from the data
if len(class_distribution) < 2:
    print("Cluster sample contains only one class. Resampling...")
    cluster_sample = data_balanced.sample(n=n, random_state=42)

print("\nUpdated Cluster Sample Class Distribution:\n", cluster_sample['Class'].value_counts())

# Reservoir Sampling
def reservoir_sampling(data, n):
    reservoir = []
    for i, row in enumerate(data.itertuples(index=False)):
        if i < n:
            reservoir.append(row)
        else:
            j = random.randint(0, i)
            if j < n:
                reservoir[j] = row
    return pd.DataFrame(reservoir, columns=data.columns)

reservoir_sample = reservoir_sampling(data, n)
print("\nReservoir Sample:\n", reservoir_sample.head())

def evaluate_models(sample, sample_name):
    X_train, X_test, y_train, y_test = train_test_split(
        sample.drop(['Class', 'Cluster'], axis=1, errors='ignore'),
        sample['Class'],
        test_size=0.3,
        random_state=42
    )

    # Define the models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        "KNN": KNeighborsClassifier()
    }

    # Evaluate each model
    print(f"\nEvaluating {sample_name}")
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.4f}")

# Evaluate each sampling technique
evaluate_models(simple_random_sample, "Simple Random Sampling")
evaluate_models(systematic_sample, "Systematic Sampling")
evaluate_models(stratified_sample, "Stratified Sampling")
evaluate_models(cluster_sample, "Cluster Sampling")
evaluate_models(reservoir_sample, "Reservoir Sampling")









