"""
predict.py
==========
Mirrors CELL 20 logic from loan_approval_prediction.py
Called by server.js via: python3 predict.py '<json>'
"""
import sys, json, pickle, os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    with open(os.path.join(BASE, name), 'rb') as f:
        return pickle.load(f)

def run(data):
    model    = load('model.pkl')
    scaler   = load('scaler.pkl')
    encoders = load('encoders.pkl')

    # ── Encode exactly like CELL 8 ──────────────────────────────────────────
    def enc(col, val):
        le = encoders[col]
        val_str = str(val)
        if val_str in le.classes_:
            return int(le.transform([val_str])[0])
        return 0   # fallback for unseen labels

    gender        = enc('Gender',        data['gender'])
    married       = enc('Married',       data['married'])
    dependents    = enc('Dependents',    str(data['dependents']))
    education     = enc('Education',     data['education'])
    self_employed = enc('Self_Employed', data['selfEmployed'])
    property_area = enc('Property_Area', data['propertyArea'])

    app_income    = float(data['applicantIncome'])
    coapp_income  = float(data['coapplicantIncome'])
    loan_amount   = float(data['loanAmount'])
    loan_term     = float(data['loanAmountTerm'])
    credit_hist   = float(data['creditHistory'])

    # ── Build feature array (same column order as CELL 10) ──────────────────
    # X.columns = ['Gender','Married','Dependents','Education','Self_Employed',
    #              'ApplicantIncome','CoapplicantIncome','LoanAmount',
    #              'Loan_Amount_Term','Credit_History','Property_Area']
    features = np.array([[
        gender, married, dependents, education, self_employed,
        app_income, coapp_income, loan_amount, loan_term,
        credit_hist, property_area
    ]])

    # ── Scale (CELL 12) ──────────────────────────────────────────────────────
    features_sc = scaler.transform(features)

    # ── Predict (CELL 14) ────────────────────────────────────────────────────
    prediction   = int(model.predict(features_sc)[0])
    probability  = model.predict_proba(features_sc)[0].tolist()

    # ── Feature importance like CELL 18 ─────────────────────────────────────
    feat_names = ['Gender','Married','Dependents','Education','Self_Employed',
                  'ApplicantIncome','CoapplicantIncome','LoanAmount',
                  'Loan_Amount_Term','Credit_History','Property_Area']
    coef = model.coef_[0].tolist()

    top_factors = sorted(
        [{"feature": feat_names[i], "coefficient": round(coef[i], 4)}
         for i in range(len(feat_names))],
        key=lambda x: abs(x["coefficient"]), reverse=True
    )[:5]

    # ── Output (mirrors CELL 20 print) ───────────────────────────────────────
    return {
        "prediction" : prediction,
        "status"     : "Approved" if prediction == 1 else "Rejected",
        "probability": {
            "approved": round(probability[1] * 100, 2),
            "rejected": round(probability[0] * 100, 2),
        },
        "topFactors" : top_factors,
    }

if __name__ == '__main__':
    try:
        result = run(json.loads(sys.argv[1]))
        print(json.dumps(result))
    except Exception as e:
        sys.stderr.write(json.dumps({"error": str(e)}))
        sys.exit(1)
