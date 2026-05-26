"""
train_model.py
==============
Waxaa laga soo qaatay: loan_approval_prediction.py (uploaded)
Marka hore RUN: python train_model.py
Waxay abuuraysaa: model.pkl, scaler.pkl, encoders.pkl, stats.json
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

# =============================================================================
# CELL 2 — Load Dataset (exact from your code)
# =============================================================================
np.random.seed(42)
n = 614

df = pd.DataFrame({
    'Loan_ID': ['LP' + str(i).zfill(6) for i in range(1, n + 1)],
    'Gender': np.random.choice(['Male', 'Female', np.nan], n, p=[0.80, 0.18, 0.02]),
    'Married': np.random.choice(['Yes', 'No', np.nan], n, p=[0.65, 0.33, 0.02]),
    'Dependents': np.random.choice(['0', '1', '2', '3+', np.nan], n, p=[0.57, 0.17, 0.16, 0.08, 0.02]),
    'Education': np.random.choice(['Graduate', 'Not Graduate'], n, p=[0.78, 0.22]),
    'Self_Employed': np.random.choice(['Yes', 'No', np.nan], n, p=[0.14, 0.81, 0.05]),
    'ApplicantIncome': np.random.randint(1500, 15000, n),
    'CoapplicantIncome': np.random.choice(
        np.concatenate([np.zeros(200), np.random.randint(1000, 7000, 414)]), n
    ),
    'LoanAmount': np.random.choice(
        np.concatenate([np.random.randint(50, 400, 600), [np.nan] * 14]), n
    ),
    'Loan_Amount_Term': np.random.choice([360, 180, 480, 300, 240, np.nan], n,
                                          p=[0.83, 0.06, 0.04, 0.03, 0.02, 0.02]),
    'Credit_History': np.random.choice([1.0, 0.0, np.nan], n, p=[0.84, 0.08, 0.08]),
    'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n, p=[0.33, 0.38, 0.29]),
    'Loan_Status': np.random.choice(['Y', 'N'], n, p=[0.69, 0.31])
})
print(f"Dataset loaded! Shape: {df.shape}")

# =============================================================================
# CELL 7 — Handle Missing Values (exact from your code)
# =============================================================================
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

# =============================================================================
# CELL 8 — Remove Duplicates & Encode (exact from your code)
# =============================================================================
df.drop_duplicates(inplace=True)
df.drop('Loan_ID', axis=1, inplace=True)

le = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education',
                    'Self_Employed', 'Property_Area', 'Loan_Status']

# Save individual encoders per column
encoders = {}
for col in categorical_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col].astype(str))
    encoders[col] = enc

# =============================================================================
# CELL 10-12 — Feature Split + Scale (exact from your code)
# =============================================================================
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# =============================================================================
# CELL 13 — Train Logistic Regression (exact from your code)
# =============================================================================
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# =============================================================================
# CELL 14-15 — Evaluate (exact from your code)
# =============================================================================
y_pred       = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_proba)

cm        = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# CELL 18 — Feature Importance
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("=" * 45)
print("   MODEL EVALUATION RESULTS")
print("=" * 45)
print(f"  Accuracy  : {accuracy  * 100:.2f}%")
print(f"  Precision : {precision * 100:.2f}%")
print(f"  Recall    : {recall    * 100:.2f}%")
print(f"  F1-Score  : {f1        * 100:.2f}%")
print(f"  ROC-AUC   : {roc_auc   * 100:.2f}%")
print("=" * 45)

# ROC curve data
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# =============================================================================
# Save all artifacts
# =============================================================================
with open('model.pkl',    'wb') as f: pickle.dump(model,    f)
with open('scaler.pkl',   'wb') as f: pickle.dump(scaler,   f)
with open('encoders.pkl', 'wb') as f: pickle.dump(encoders, f)

stats = {
    "accuracy"       : round(float(accuracy),  4),
    "precision"      : round(float(precision), 4),
    "recall"         : round(float(recall),    4),
    "f1"             : round(float(f1),        4),
    "roc_auc"        : round(float(roc_auc),   4),
    "total_samples"  : int(len(df)),
    "approved_count" : int(y.sum()),
    "rejected_count" : int(len(y) - y.sum()),
    "approval_rate"  : round(float(y.mean()), 4),
    "feature_names"  : list(X.columns),
    "coefficients"   : [round(float(c), 6) for c in model.coef_[0]],
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "roc_curve"      : {
        "fpr": [round(float(v), 4) for v in fpr[:50]],
        "tpr": [round(float(v), 4) for v in tpr[:50]],
    },
    "metrics_summary": [
        {"name": "Accuracy",  "value": round(float(accuracy)  * 100, 2)},
        {"name": "Precision", "value": round(float(precision) * 100, 2)},
        {"name": "Recall",    "value": round(float(recall)    * 100, 2)},
        {"name": "F1-Score",  "value": round(float(f1)        * 100, 2)},
        {"name": "ROC-AUC",   "value": round(float(roc_auc)   * 100, 2)},
    ]
}
with open('stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\n✅  Saved: model.pkl | scaler.pkl | encoders.pkl | stats.json")
print(f"   Ready for API at http://localhost:5000")
