#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:29:14 2026

@author: root1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:01:33 2026

@author: root1
"""

# Step 1: Import libraries
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV

path = '/home/root1/'
os.chdir(path)
sys.path.append(path)


# =========================
# Step 1: Load Data
# =========================
df = pd.read_csv("INCART 2-lead Arrhythmia Database.csv", skiprows=1, header=None)

# =========================
# Step 2: Target Variable
# =========================
# Column 1 contains labels (N, VEB, etc.)
y = df.iloc[:, 1]

# Convert to binary:
# 0 = Normal, 1 = Abnormal
y = y.apply(lambda x: 0 if x == 'N' else 1)

# =========================
# Step 3: Features
# =========================
# Remove ID + label columns
X = df.iloc[:, 2:]

# =========================
# Step 4: Train-Test Split FIRST (NO LEAKAGE)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Step 5: Preprocessing (FIT ONLY ON TRAIN)
# =========================
imputer = SimpleImputer(strategy='mean')

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# =========================
# Step 6: PCA (FIT ONLY ON TRAIN)
# =========================
pca = PCA(n_components=0.95, random_state=42)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Reduced features: {X_train_scaled.shape[1]} → {X_train_pca.shape[1]}")

# =========================
# Step 7: Model + GridSearch
# =========================
svm = SVC(kernel='rbf')

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001]
}

grid = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_pca, y_train)

print("Best Parameters:", grid.best_params_)

model = grid.best_estimator_

# =========================
# Step 8: Evaluation
# =========================
y_pred = model.predict(X_test_pca)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# Step 9: Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =========================
# Step 10: Class Distribution
# =========================
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Class Distribution (0 = Normal, 1 = Abnormal)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =========================
# Optional: ROC-AUC (BONUS)
# =========================
from sklearn.metrics import roc_auc_score

y_prob = model.decision_function(X_test_pca)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:", roc_auc)

SVC(class_weight='balanced')