/**
 * server.js — Node.js / Express + MongoDB backend
 *
 * Routes:
 *   POST   /api/predict           → calls ML service, saves to MongoDB
 *   GET    /api/predictions        → all past predictions from MongoDB
 *   GET    /api/predictions/:id    → single prediction
 *   DELETE /api/predictions/:id    → delete a record
 *   GET    /api/stats              → ML model stats
 *   GET    /api/db-stats           → MongoDB summary counts
 *   POST   /api/retrain            → retrain model
 *   GET    /api/health             → health check
 */

const express  = require('express');
const cors     = require('cors');
const mongoose = require('mongoose');
const axios    = require('axios');

const app    = express();
const PORT   = process.env.PORT     || 3500;
const ML_URL = process.env.ML_URL   || 'http://localhost:5001';
const MONGO  = process.env.MONGO_URI || 'mongodb://localhost:27017/loaniq';

app.use(cors({ origin: '*' }));
app.use(express.json());

// MongoDB
mongoose.connect(MONGO)
  .then(() => console.log('MongoDB connected:', MONGO))
  .catch(err => console.error('MongoDB error:', err.message));

// Schema
const predictionSchema = new mongoose.Schema({
  gender: String, married: String, dependents: String,
  education: String, selfEmployed: String,
  applicantIncome: Number, coapplicantIncome: Number,
  loanAmount: Number, loanAmountTerm: Number,
  creditHistory: Number, propertyArea: String,
  prediction: Number,
  status: String,
  probability: { approved: Number, rejected: Number },
  topFactors: [{ feature: String, coefficient: Number }],
}, { timestamps: true });

const Prediction = mongoose.model('Prediction', predictionSchema);

// Health
app.get('/api/health', async (req, res) => {
  try {
    await axios.get(`${ML_URL}/health`);
    res.json({ node: 'ok', ml: 'ok', mongo: mongoose.connection.readyState === 1 ? 'ok' : 'disconnected' });
  } catch {
    res.json({ node: 'ok', ml: 'unreachable', mongo: mongoose.connection.readyState === 1 ? 'ok' : 'disconnected' });
  }
});

// ML stats
app.get('/api/stats', async (req, res) => {
  try {
    const { data } = await axios.get(`${ML_URL}/stats`);
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: 'ML service unreachable: ' + err.message });
  }
});

// DB stats
app.get('/api/db-stats', async (req, res) => {
  try {
    const total    = await Prediction.countDocuments();
    const approved = await Prediction.countDocuments({ status: 'Approved' });
    const rejected = await Prediction.countDocuments({ status: 'Rejected' });
    const latest   = await Prediction.findOne().sort({ createdAt: -1 });
    const agg      = await Prediction.aggregate([{ $group: { _id: null, avg: { $avg: '$applicantIncome' } } }]);
    res.json({
      total, approved, rejected,
      approvalRate: total ? ((approved / total) * 100).toFixed(1) : 0,
      latestPrediction: latest,
      avgApplicantIncome: agg[0]?.avg?.toFixed(0) || 0,
    });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// Predict
app.post('/api/predict', async (req, res) => {
  const fields = ['gender','married','dependents','education','selfEmployed',
    'applicantIncome','coapplicantIncome','loanAmount','loanAmountTerm','creditHistory','propertyArea'];
  for (const f of fields) {
    if (req.body[f] === undefined || req.body[f] === '')
      return res.status(400).json({ error: `Missing field: ${f}` });
  }
  try {
    const { data: ml } = await axios.post(`${ML_URL}/predict`, req.body);
    if (ml.error) return res.status(500).json({ error: ml.error });
    const doc = await Prediction.create({
      ...req.body,
      prediction: ml.prediction, status: ml.status,
      probability: ml.probability, topFactors: ml.topFactors,
    });
    res.json({ ...ml, _id: doc._id, savedAt: doc.createdAt });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// List predictions
app.get('/api/predictions', async (req, res) => {
  try {
    const page  = parseInt(req.query.page  || 1);
    const limit = parseInt(req.query.limit || 10);
    const filter = {};
    if (req.query.status) filter.status = req.query.status;
    const [docs, total] = await Promise.all([
      Prediction.find(filter).sort({ createdAt: -1 }).skip((page-1)*limit).limit(limit),
      Prediction.countDocuments(filter),
    ]);
    res.json({ data: docs, total, page, pages: Math.ceil(total / limit) });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// Single prediction
app.get('/api/predictions/:id', async (req, res) => {
  try {
    const doc = await Prediction.findById(req.params.id);
    if (!doc) return res.status(404).json({ error: 'Not found' });
    res.json(doc);
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// Delete
app.delete('/api/predictions/:id', async (req, res) => {
  try {
    await Prediction.findByIdAndDelete(req.params.id);
    res.json({ message: 'Deleted successfully' });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// Retrain
app.post('/api/retrain', async (req, res) => {
  try {
    const { data } = await axios.post(`${ML_URL}/retrain`);
    res.json(data);
  } catch (err) { res.status(500).json({ error: err.message }); }
});

app.listen(PORT, () => {
  console.log(`Node server  → http://localhost:${PORT}`);
  console.log(`ML service   → ${ML_URL}`);
  console.log(`MongoDB      → ${MONGO}`);
});
