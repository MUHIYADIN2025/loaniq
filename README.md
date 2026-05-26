# 🏦 LoanIQ — Full Stack ML System
## React + Node.js + MongoDB + Python ML

---

## 📁 Project Structure

```
loaniq/
├── backend/
│   ├── ml_service.py     ← Python Flask ML (port 5001)
│   ├── server.js         ← Node.js Express + MongoDB (port 5000)
│   └── package.json
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── api.js             ← All API calls
    │   ├── index.css
    │   ├── main.jsx
    │   └── pages/
    │       ├── Predict.jsx    ← Loan form + live result
    │       ├── Dashboard.jsx  ← Charts + ML + MongoDB stats
    │       └── History.jsx    ← All past predictions (CRUD)
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## 🚀 Setup — 5 Steps

### Step 1 — Install Python dependencies
```bash
pip install flask flask-cors pandas numpy scikit-learn
```

### Step 2 — Start Python ML service
```bash
cd backend
python3 ml_service.py
# Runs on http://localhost:5001
# Trains model automatically on first run
```

### Step 3 — Start MongoDB
```bash
# Make sure MongoDB is installed and running:
mongod --dbpath /data/db
# OR if using brew:
brew services start mongodb-community
```

### Step 4 — Start Node.js backend
```bash
cd backend
npm install
node server.js
# Runs on http://localhost:5000
```

### Step 5 — Start React frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## 🔌 API Reference

| Method | Endpoint                | Description                  |
|--------|-------------------------|------------------------------|
| GET    | /api/health             | Check all services           |
| GET    | /api/stats              | ML model metrics             |
| GET    | /api/db-stats           | MongoDB summary              |
| POST   | /api/predict            | Run prediction + save to DB  |
| GET    | /api/predictions        | All past predictions         |
| DELETE | /api/predictions/:id    | Delete a record              |
| POST   | /api/retrain            | Retrain the ML model         |

---

## 📄 Pages

| Page        | Route        | Features                                     |
|-------------|--------------|----------------------------------------------|
| Predict     | /            | Form, instant result, probability bars, top factors |
| Dashboard   | /dashboard   | 8 stat cards, 6 charts, confusion matrix     |
| History     | /history     | Table, filters, pagination, expand, delete   |

---

## 🛠 Tech Stack

| Layer     | Technology                        |
|-----------|-----------------------------------|
| Frontend  | React 18, Vite, Recharts          |
| Backend   | Node.js, Express, Mongoose        |
| Database  | MongoDB                           |
| ML        | Python, Flask, scikit-learn       |
| Algorithm | Logistic Regression               |
