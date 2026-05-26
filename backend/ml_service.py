"""
ml_service.py  —  Flask ML microservice
- Trains the Logistic Regression model from loan_approval_prediction.py
- Exposes POST /predict  and  GET /stats
- Every prediction is saved to MongoDB via the Node backend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle, os, json, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

app   = Flask(__name__)
CORS(app)
BASE  = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# TRAIN  (mirrors your loan_approval_prediction.py exactly)
# ──────────────────────────────────────────────
def train():
    np.random.seed(42)
    n = 614

    df = pd.DataFrame({
        'Loan_ID'          : ['LP' + str(i).zfill(6) for i in range(1, n + 1)],
        'Gender'           : np.random.choice(['Male','Female',np.nan], n, p=[0.80,0.18,0.02]),
        'Married'          : np.random.choice(['Yes','No',np.nan],      n, p=[0.65,0.33,0.02]),
        'Dependents'       : np.random.choice(['0','1','2','3+',np.nan],n, p=[0.57,0.17,0.16,0.08,0.02]),
        'Education'        : np.random.choice(['Graduate','Not Graduate'],n, p=[0.78,0.22]),
        'Self_Employed'    : np.random.choice(['Yes','No',np.nan],      n, p=[0.14,0.81,0.05]),
        'ApplicantIncome'  : np.random.randint(1500, 15000, n),
        'CoapplicantIncome': np.random.choice(
            np.concatenate([np.zeros(200), np.random.randint(1000,7000,414)]), n),
        'LoanAmount'       : np.random.choice(
            np.concatenate([np.random.randint(50,400,600), [np.nan]*14]), n),
        'Loan_Amount_Term' : np.random.choice([360,180,480,300,240,np.nan], n,
                                               p=[0.83,0.06,0.04,0.03,0.02,0.02]),
        'Credit_History'   : np.random.choice([1.0,0.0,np.nan], n, p=[0.84,0.08,0.08]),
        'Property_Area'    : np.random.choice(['Urban','Semiurban','Rural'], n, p=[0.33,0.38,0.29]),
        'Loan_Status'      : np.random.choice(['Y','N'], n, p=[0.69,0.31]),
    })

    # — Missing values
    df['LoanAmount'].fillna(df['LoanAmount'].mean(),             inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    for col in ['Gender','Married','Dependents','Self_Employed']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df.drop_duplicates(inplace=True)
    df.drop('Loan_ID', axis=1, inplace=True)

    # — Encode
    encoders = {}
    for col in ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Ensure no NaN values remain before training
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)

    model  = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_tr, y_train)

    y_pred       = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    stats = {
        'accuracy'      : float(accuracy_score(y_test, y_pred)),
        'precision'     : float(precision_score(y_test, y_pred)),
        'recall'        : float(recall_score(y_test, y_pred)),
        'f1'            : float(f1_score(y_test, y_pred)),
        'roc_auc'       : float(roc_auc_score(y_test, y_pred_proba)),
        'total_samples' : int(len(df)),
        'approved_count': int(y.sum()),
        'rejected_count': int(len(y) - y.sum()),
        'feature_names' : list(X.columns),
        'coefficients'  : model.coef_[0].tolist(),
        'confusion'     : { 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp) },
        'class_report'  : classification_report(y_test, y_pred,
                              target_names=['Rejected','Approved'], output_dict=True),
    }

    with open(os.path.join(BASE,'model.pkl'),    'wb') as f: pickle.dump(model,    f)
    with open(os.path.join(BASE,'scaler.pkl'),   'wb') as f: pickle.dump(scaler,   f)
    with open(os.path.join(BASE,'encoders.pkl'), 'wb') as f: pickle.dump(encoders, f)
    with open(os.path.join(BASE,'stats.json'),   'w')  as f: json.dump(stats, f, indent=2)

    print(f"✅ Model trained — Accuracy: {stats['accuracy']*100:.2f}%")
    return model, scaler, encoders, stats

# ──────────────────────────────────────────────
# LOAD or train
# ──────────────────────────────────────────────
if all(os.path.exists(os.path.join(BASE, f))
       for f in ['model.pkl','scaler.pkl','encoders.pkl','stats.json']):
    with open(os.path.join(BASE,'model.pkl'),    'rb') as f: MODEL    = pickle.load(f)
    with open(os.path.join(BASE,'scaler.pkl'),   'rb') as f: SCALER   = pickle.load(f)
    with open(os.path.join(BASE,'encoders.pkl'), 'rb') as f: ENCODERS = pickle.load(f)
    with open(os.path.join(BASE,'stats.json'))         as f: STATS    = json.load(f)
    print("✅ Loaded existing model artifacts")
else:
    MODEL, SCALER, ENCODERS, STATS = train()

FEATURE_NAMES = ['Gender','Married','Dependents','Education','Self_Employed',
                 'ApplicantIncome','CoapplicantIncome','LoanAmount',
                 'Loan_Amount_Term','Credit_History','Property_Area']

def encode(field, value):
    le = ENCODERS.get(field)
    if le and value in le.classes_:
        return int(le.transform([value])[0])
    return 0

# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'ML Service'})

@app.get('/stats')
def stats():
    return jsonify(STATS)

@app.post('/predict')
def predict():
    d = request.json
    try:
        features = np.array([[
            encode('Gender',       d.get('gender',       'Male')),
            encode('Married',      d.get('married',      'Yes')),
            int(str(d.get('dependents', 0)).replace('3+', '3')),
            encode('Education',    d.get('education',    'Graduate')),
            encode('Self_Employed',d.get('selfEmployed', 'No')),
            float(d.get('applicantIncome',   0)),
            float(d.get('coapplicantIncome', 0)),
            float(d.get('loanAmount',        100)),
            float(d.get('loanAmountTerm',    360)),
            float(d.get('creditHistory',     1)),
            encode('Property_Area',d.get('propertyArea', 'Urban')),
        ]])

        sc_feat  = SCALER.transform(features)
        pred     = int(MODEL.predict(sc_feat)[0])
        proba    = MODEL.predict_proba(sc_feat)[0].tolist()
        coefs    = MODEL.coef_[0].tolist()

        factors = sorted([
            {'feature': FEATURE_NAMES[i], 'coefficient': round(coefs[i], 4)}
            for i in range(len(FEATURE_NAMES))
        ], key=lambda x: abs(x['coefficient']), reverse=True)[:5]

        return jsonify({
            'prediction' : pred,
            'status'     : 'Approved' if pred == 1 else 'Rejected',
            'probability': {
                'approved': round(proba[1] * 100, 2),
                'rejected': round(proba[0] * 100, 2),
            },
            'topFactors' : factors,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.post('/retrain')
def retrain():
    global MODEL, SCALER, ENCODERS, STATS
    MODEL, SCALER, ENCODERS, STATS = train()
    return jsonify({'message': 'Model retrained successfully', 'accuracy': STATS['accuracy']})

if __name__ == '__main__':
    app.run(port=5001, debug=False)
