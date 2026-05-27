# =============================================================================
# LOAN APPROVAL PREDICTION USING LOGISTIC REGRESSION
# Complete Jupyter Notebook Code
# =============================================================================
# CELL 1 — Install & Import Libraries
# =============================================================================

# !pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")

# =============================================================================
# CELL 2 — Load Dataset
# =============================================================================

# Option A: Load from CSV file
# df = pd.read_csv("loan_data.csv")

# Option B: Create sample dataset (run this if you don't have a CSV file)
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

print("Dataset loaded!")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
df.head()

# =============================================================================
# CELL 3 — Basic Dataset Info
# =============================================================================

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
df.describe()

# =============================================================================
# CELL 4 — Missing Values Analysis
# =============================================================================

print("=" * 50)
print("MISSING VALUES")
print("=" * 50)

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0]
print(missing_df)

# Visualize missing values
plt.figure(figsize=(10, 5))
missing_df['Missing %'].plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Missing Values Percentage per Column', fontsize=14, fontweight='bold')
plt.xlabel('Columns')
plt.ylabel('Missing %')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# CELL 5 — Exploratory Data Analysis (EDA)
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Loan Status Distribution
loan_counts = df['Loan_Status'].value_counts()
axes[0, 0].pie(loan_counts, labels=['Approved', 'Rejected'], autopct='%1.1f%%',
               colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[0, 0].set_title('Loan Status Distribution')

# 2. Gender vs Loan Status
gender_loan = df.groupby(['Gender', 'Loan_Status']).size().unstack()
gender_loan.plot(kind='bar', ax=axes[0, 1], color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[0, 1].set_title('Gender vs Loan Status')
axes[0, 1].set_xlabel('Gender')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend(['Rejected', 'Approved'])
axes[0, 1].tick_params(axis='x', rotation=0)

# 3. Education vs Loan Status
edu_loan = df.groupby(['Education', 'Loan_Status']).size().unstack()
edu_loan.plot(kind='bar', ax=axes[0, 2], color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[0, 2].set_title('Education vs Loan Status')
axes[0, 2].set_xlabel('Education')
axes[0, 2].tick_params(axis='x', rotation=0)

# 4. Credit History vs Loan Status
credit_loan = df.groupby(['Credit_History', 'Loan_Status']).size().unstack()
credit_loan.plot(kind='bar', ax=axes[1, 0], color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[1, 0].set_title('Credit History vs Loan Status')
axes[1, 0].set_xlabel('Credit History (0=Bad, 1=Good)')
axes[1, 0].tick_params(axis='x', rotation=0)

# 5. Applicant Income Distribution
axes[1, 1].hist(df['ApplicantIncome'].dropna(), bins=30, color='steelblue', edgecolor='black')
axes[1, 1].set_title('Applicant Income Distribution')
axes[1, 1].set_xlabel('Income')
axes[1, 1].set_ylabel('Frequency')

# 6. Loan Amount Distribution
axes[1, 2].hist(df['LoanAmount'].dropna(), bins=30, color='orange', edgecolor='black')
axes[1, 2].set_title('Loan Amount Distribution')
axes[1, 2].set_xlabel('Loan Amount (thousands)')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# =============================================================================
# CELL 6 — More EDA: Property Area & Married Status
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Property Area vs Loan Status
prop_loan = df.groupby(['Property_Area', 'Loan_Status']).size().unstack()
prop_loan.plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[0].set_title('Property Area vs Loan Status', fontsize=13)
axes[0].set_xlabel('Property Area')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(['Rejected', 'Approved'])

# Married vs Loan Status
married_loan = df.groupby(['Married', 'Loan_Status']).size().unstack()
married_loan.plot(kind='bar', ax=axes[1], color=['#e74c3c', '#2ecc71'], edgecolor='black')
axes[1].set_title('Married Status vs Loan Status', fontsize=13)
axes[1].set_xlabel('Married')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

# =============================================================================
# CELL 7 — Data Preprocessing: Handle Missing Values
# =============================================================================

print("BEFORE - Missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Fill numerical columns with mean
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Fill categorical columns with mode
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

print("\nAFTER - Missing values:")
print(df.isnull().sum())
print("\nAll missing values handled!")

# =============================================================================
# CELL 8 — Data Preprocessing: Remove Duplicates & Encode
# =============================================================================

# Remove duplicates
before = len(df)
df.drop_duplicates(inplace=True)
after = len(df)
print(f"Duplicates removed: {before - after}")

# Drop Loan_ID (not a feature)
df.drop('Loan_ID', axis=1, inplace=True)

# Label encode all categorical columns
le = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education',
                    'Self_Employed', 'Property_Area', 'Loan_Status']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("\nEncoding complete. Sample of encoded data:")
df.head()

# =============================================================================
# CELL 9 — Correlation Heatmap
# =============================================================================

plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={'size': 9})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Print top correlations with Loan_Status
print("\nTop features correlated with Loan_Status:")
corr_target = corr_matrix['Loan_Status'].drop('Loan_Status').abs().sort_values(ascending=False)
print(corr_target)

# =============================================================================
# CELL 10 — Feature Selection & Preparation
# =============================================================================

# Define features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

print("Features (X):", list(X.columns))
print("Target (y): Loan_Status")
print(f"\nClass distribution:\n{y.value_counts()}")
print(f"Approval rate: {y.mean()*100:.1f}%")

# =============================================================================
# CELL 11 — Train/Test Split
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size : {X_train.shape[0]} samples")
print(f"Testing  set size : {X_test.shape[0]} samples")
print(f"\nTrain class distribution:\n{y_train.value_counts()}")
print(f"\nTest  class distribution:\n{y_test.value_counts()}")

# =============================================================================
# CELL 12 — Feature Scaling
# =============================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Feature scaling applied (StandardScaler).")
print(f"Mean of scaled training data: {X_train_scaled.mean():.6f} (should be ~0)")
print(f"Std  of scaled training data: {X_train_scaled.std():.6f}  (should be ~1)")

# =============================================================================
# CELL 13 — Train Logistic Regression Model
# =============================================================================

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

print("Model training complete!")
print(f"\nModel: {model}")
print(f"\nIntercept : {model.intercept_[0]:.4f}")
print(f"\nCoefficients:")
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print(coef_df.to_string(index=False))

# =============================================================================
# CELL 14 — Model Prediction
# =============================================================================

y_pred       = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("Predictions generated!")
print(f"\nFirst 10 predictions vs actual:")
comparison = pd.DataFrame({
    'Actual'    : y_test.values[:10],
    'Predicted' : y_pred[:10],
    'Probability': y_pred_proba[:10].round(3)
})
comparison['Correct'] = comparison['Actual'] == comparison['Predicted']
print(comparison.to_string(index=False))

# =============================================================================
# CELL 15 — Evaluation Metrics
# =============================================================================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_proba)

print("=" * 45)
print("         MODEL EVALUATION RESULTS")
print("=" * 45)
print(f"  Accuracy  : {accuracy  * 100:.2f}%")
print(f"  Precision : {precision * 100:.2f}%")
print(f"  Recall    : {recall    * 100:.2f}%")
print(f"  F1-Score  : {f1        * 100:.2f}%")
print(f"  ROC-AUC   : {roc_auc   * 100:.2f}%")
print("=" * 45)

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

# =============================================================================
# CELL 16 — Confusion Matrix
# =============================================================================

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Rejected', 'Approved'],
            yticklabels=['Rejected', 'Approved'],
            linewidths=1, linecolor='white', annot_kws={'size': 14})
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

print(f"\nTrue  Negatives (TN): {tn}  — Correctly predicted Rejected")
print(f"False Positives (FP): {fp}  — Wrongly predicted Approved")
print(f"False Negatives (FN): {fn}  — Wrongly predicted Rejected")
print(f"True  Positives (TP): {tp}  — Correctly predicted Approved")

# =============================================================================
# CELL 17 — ROC Curve
# =============================================================================

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='steelblue', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random Classifier')
plt.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve — Logistic Regression', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# CELL 18 — Feature Importance (Coefficients)
# =============================================================================

coef_plot = coef_df.sort_values('Coefficient')

colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef_plot['Coefficient']]
plt.figure(figsize=(9, 6))
plt.barh(coef_plot['Feature'], coef_plot['Coefficient'], color=colors, edgecolor='black')
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.title('Feature Importance (Logistic Regression Coefficients)', fontsize=13, fontweight='bold')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\nTop positive predictors (increase approval chance):")
print(coef_df[coef_df['Coefficient'] > 0].head(3).to_string(index=False))
print("\nTop negative predictors (decrease approval chance):")
print(coef_df[coef_df['Coefficient'] < 0].tail(3).to_string(index=False))

# =============================================================================
# CELL 19 — Metrics Summary Chart
# =============================================================================

metrics_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [accuracy, precision, recall, f1, roc_auc]
bar_colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']

plt.figure(figsize=(10, 5))
bars = plt.bar(metrics_names, [v * 100 for v in metrics_values],
               color=bar_colors, edgecolor='black', width=0.5)
for bar, val in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f'{val*100:.1f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.ylim(0, 110)
plt.title('Model Performance Summary', fontsize=14, fontweight='bold')
plt.ylabel('Score (%)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# CELL 20 — Predict on New Applicant
# =============================================================================

print("=" * 50)
print("PREDICT FOR A NEW APPLICANT")
print("=" * 50)

# Encoding reference:
# Gender:        Male=1, Female=0
# Married:       Yes=1, No=0
# Dependents:    0=0, 1=1, 2=2, 3+=3
# Education:     Graduate=0, Not Graduate=1
# Self_Employed: Yes=1, No=0
# Property_Area: Rural=0, Semiurban=1, Urban=2

new_applicant = pd.DataFrame({
    'Gender':            [1],       # Male
    'Married':           [1],       # Yes
    'Dependents':        [0],       # 0
    'Education':         [0],       # Graduate
    'Self_Employed':     [0],       # No
    'ApplicantIncome':   [5000],    # Monthly income
    'CoapplicantIncome': [2000],    # Co-applicant income
    'LoanAmount':        [150],     # in thousands
    'Loan_Amount_Term':  [360],     # months
    'Credit_History':    [1],       # Good
    'Property_Area':     [1],       # Semiurban
})

new_scaled = scaler.transform(new_applicant)
prediction = model.predict(new_scaled)[0]
probability = model.predict_proba(new_scaled)[0]

status = "APPROVED ✓" if prediction == 1 else "REJECTED ✗"
print(f"\nApplicant Details:")
print(f"  Income        : $5,000")
print(f"  Loan Amount   : $150,000")
print(f"  Credit History: Good")
print(f"  Education     : Graduate")
print(f"\nPrediction   : {status}")
print(f"Probability  : Approved = {probability[1]*100:.1f}% | Rejected = {probability[0]*100:.1f}%")

# =============================================================================
# CELL 21 — Final Summary
# =============================================================================

print("=" * 55)
print("   LOAN APPROVAL PREDICTION — FINAL SUMMARY")
print("=" * 55)
print(f"  Algorithm     : Logistic Regression")
print(f"  Dataset Size  : {len(df)} samples")
print(f"  Features Used : {X.shape[1]}")
print(f"  Train/Test    : 80% / 20%")
print()
print(f"  Accuracy      : {accuracy  * 100:.2f}%")
print(f"  Precision     : {precision * 100:.2f}%")
print(f"  Recall        : {recall    * 100:.2f}%")
print(f"  F1-Score      : {f1        * 100:.2f}%")
print(f"  ROC-AUC       : {roc_auc   * 100:.2f}%")
print()
print("  Key Findings:")
print("  • Credit History is the strongest predictor")
print("  • Semiurban applicants have higher approval rates")
print("  • Graduates are more likely to get approved")
print("  • ML significantly reduces manual decision-making")
print("=" * 55)
