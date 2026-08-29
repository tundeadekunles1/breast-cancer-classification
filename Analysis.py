"""
Breast Cancer Classification
Predicting whether a tumor is malignant (M) or benign (B) from
diagnostic measurements, using the UCI Breast Cancer Wisconsin
(Diagnostic) dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)

# ---------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------
df = pd.read_csv("data/breast-cancer.csv")

print("Dataset shape:", df.shape)
print("\nClass balance:")
print(df["diagnosis"].value_counts())

# Encode target: M (malignant) -> 1, B (benign) -> 0
df["target"] = df["diagnosis"].map({"M": 1, "B": 0})

feature_cols = [c for c in df.columns if c not in ["id", "diagnosis", "target"]]
X = df[feature_cols]
y = df["target"]

# ---------------------------------------------------------------
# 2. Train/test split + scaling
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 3. Train models
# ---------------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)  # tree models don't need scaling

models = {"Logistic Regression": (log_reg, X_test_scaled), "Random Forest": (rf, X_test)}

# ---------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------
results = {}
for name, (model, X_te) in models.items():
    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
        "preds": preds,
        "proba": proba,
    }
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {results[name]['accuracy']:.3f}")
    print(f"Precision: {results[name]['precision']:.3f}")
    print(f"Recall:    {results[name]['recall']:.3f}")
    print(f"F1 score:  {results[name]['f1']:.3f}")
    print(f"ROC-AUC:   {results[name]['roc_auc']:.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

# ---------------------------------------------------------------
# 5. Visualizations
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 5a. Class balance
df["diagnosis"].value_counts().plot(kind="bar", ax=axes[0], color=["#4C72B0", "#C44E52"])
axes[0].set_title("Class Balance (B = Benign, M = Malignant)")
axes[0].set_xlabel("Diagnosis")
axes[0].set_ylabel("Count")

# 5b. Feature importance (Random Forest)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
importances.plot(kind="barh", ax=axes[1], color="#55A868")
axes[1].invert_yaxis()
axes[1].set_title("Top 10 Feature Importances (Random Forest)")
axes[1].set_xlabel("Importance")

# 5c. ROC curves
for name, (model, X_te) in models.items():
    fpr, tpr, _ = roc_curve(y_test, results[name]["proba"])
    axes[2].plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[2].set_title("ROC Curves")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].legend(loc="lower right")

plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("\nSaved visualizations to results.png")

# ---------------------------------------------------------------
# 6. Save a summary for the README
# ---------------------------------------------------------------
with open("metrics_summary.txt", "w") as f:
    f.write(f"Dataset size: {df.shape[0]} samples, {len(feature_cols)} features\n")
    f.write(f"Class balance: {dict(df['diagnosis'].value_counts())}\n\n")
    for name in results:
        f.write(f"{name}:\n")
        f.write(f"  Accuracy:  {results[name]['accuracy']:.3f}\n")
        f.write(f"  Precision: {results[name]['precision']:.3f}\n")
        f.write(f"  Recall:    {results[name]['recall']:.3f}\n")
        f.write(f"  F1 score:  {results[name]['f1']:.3f}\n")
        f.write(f"  ROC-AUC:   {results[name]['roc_auc']:.3f}\n\n")

print("Saved metrics summary to metrics_summary.txt")
