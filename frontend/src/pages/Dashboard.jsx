import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
} from 'recharts'
import { api } from '../api'

const TT = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background:'#0E1520', border:'1px solid rgba(255,255,255,.1)', borderRadius:8, padding:'10px 14px', fontSize:12 }}>
      <div style={{ color:'#5A7090', marginBottom:4 }}>{label}</div>
      {payload.map((p,i)=>(
        <div key={i} style={{ color:p.color||'#E8EDF5' }}>
          {p.name}: <strong>{typeof p.value==='number' ? p.value.toFixed(3) : p.value}</strong>
        </div>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const [ml,    setMl]    = useState(null)
  const [db,    setDb]    = useState(null)
  const [loading,setLoading]= useState(true)
  const [err,   setErr]   = useState('')
  const [retraining, setRetraining] = useState(false)

  const load = async () => {
    setLoading(true); setErr('')
    try {
      const [m, d] = await Promise.all([api.stats(), api.dbStats()])
      setMl(m); setDb(d)
    } catch(e) { setErr(e.message) }
    finally { setLoading(false) }
  }

  useEffect(()=>{ load() }, [])

  const retrain = async () => {
    setRetraining(true)
    try { await api.retrain(); await load() }
    catch(e) { setErr(e.message) }
    finally { setRetraining(false) }
  }

  if (loading) return (
    <div style={{ textAlign:'center', paddingTop:100 }}>
      <div className="spinner" style={{ width:32, height:32, borderWidth:3, borderTopColor:'var(--amber)' }} />
      <div style={{ color:'var(--muted)', marginTop:16 }}>Loading dashboard...</div>
    </div>
  )

  if (err) return (
    <div>
      <div className="page-head"><h1>Dashboard</h1></div>
      <div className="err">⚠️ {err}</div>
    </div>
  )

  // ── Chart data ────────────────────────────────────────────────────────────
  const coefData = ml.feature_names.map((n,i)=>({
    name:n, value:parseFloat(ml.coefficients[i].toFixed(4)),
    abs: Math.abs(ml.coefficients[i])
  })).sort((a,b)=>b.abs-a.abs)

  const barData = [...coefData].sort((a,b)=>a.value-b.value)

  const pieMl  = [
    { name:'Rejected', value: ml.rejected_count },
    { name:'Approved', value: ml.approved_count },
  ]
  const pieDb  = db.total ? [
    { name:'Rejected', value: db.rejected },
    { name:'Approved', value: db.approved },
  ] : null

  const radarData = coefData.slice(0,7).map(d=>({
    feature: d.name, importance: parseFloat((d.abs*100).toFixed(1))
  }))

  const metrics = [
    { k:'Accuracy',  v: ml.accuracy },
    { k:'Precision', v: ml.precision },
    { k:'Recall',    v: ml.recall },
    { k:'F1-Score',  v: ml.f1 },
    { k:'ROC-AUC',   v: ml.roc_auc },
  ]
  const barMetrics = metrics.map(m=>({ name:m.k, value:parseFloat((m.v*100).toFixed(1)) }))

  return (
    <div>
      <div className="page-head" style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-end' }}>
        <div>
          <h1>Model <em>Dashboard</em></h1>
          <p>Logistic Regression performance + MongoDB database stats.</p>
        </div>
        <button className="btn btn-ghost" onClick={retrain} disabled={retraining}>
          {retraining ? <><span className="spinner" style={{ borderTopColor:'var(--amber)' }}/> Retraining...</> : '🔄 Retrain Model'}
        </button>
      </div>

      {/* ── ML stat cards ── */}
      <div className="stat-grid">
        <div className="stat-card amber">
          <div className="stat-label">Accuracy</div>
          <div className="stat-val">{(ml.accuracy*100).toFixed(1)}%</div>
          <div className="stat-sub">Model performance</div>
        </div>
        <div className="stat-card green">
          <div className="stat-label">ML Approved (dataset)</div>
          <div className="stat-val">{ml.approved_count}</div>
          <div className="stat-sub">{(ml.approved_count/ml.total_samples*100).toFixed(1)}% approval rate</div>
        </div>
        <div className="stat-card red">
          <div className="stat-label">ML Rejected (dataset)</div>
          <div className="stat-val">{ml.rejected_count}</div>
          <div className="stat-sub">{(ml.rejected_count/ml.total_samples*100).toFixed(1)}% rejection rate</div>
        </div>
        <div className="stat-card blue">
          <div className="stat-label">Total Samples</div>
          <div className="stat-val">{ml.total_samples}</div>
          <div className="stat-sub">Training dataset size</div>
        </div>
      </div>

      {/* ── MongoDB stat cards ── */}
      <div style={{ marginBottom:8, fontSize:11, color:'var(--muted)', textTransform:'uppercase', letterSpacing:'.8px' }}>
        🍃 MongoDB — Live Predictions
      </div>
      <div className="stat-grid" style={{ marginBottom:28 }}>
        <div className="stat-card teal">
          <div className="stat-label">Total Predictions</div>
          <div className="stat-val">{db.total}</div>
          <div className="stat-sub">Saved in MongoDB</div>
        </div>
        <div className="stat-card green">
          <div className="stat-label">Approved</div>
          <div className="stat-val">{db.approved}</div>
          <div className="stat-sub">{db.approvalRate}% rate</div>
        </div>
        <div className="stat-card red">
          <div className="stat-label">Rejected</div>
          <div className="stat-val">{db.rejected}</div>
          <div className="stat-sub">from live predictions</div>
        </div>
        <div className="stat-card amber">
          <div className="stat-label">Avg Income</div>
          <div className="stat-val">${Number(db.avgApplicantIncome).toLocaleString()}</div>
          <div className="stat-sub">of all applicants</div>
        </div>
      </div>

      {/* ── Charts row 1 ── */}
      <div className="charts-grid">
        {/* Metrics bar */}
        <div className="card">
          <div className="chart-title">Performance Metrics</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={barMetrics}>
              <XAxis dataKey="name" tick={{ fill:'#5A7090', fontSize:11 }} />
              <YAxis tick={{ fill:'#5A7090', fontSize:11 }} domain={[0,100]} />
              <Tooltip content={<TT />} />
              <Bar dataKey="value" radius={[4,4,0,0]}>
                {barMetrics.map((_,i)=>(
                  <Cell key={i} fill={['#F59E0B','#10B981','#3B82F6','#8B5CF6','#F43F5E'][i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Dataset pie */}
        <div className="card">
          <div className="chart-title">Dataset Distribution</div>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={pieMl} dataKey="value" cx="50%" cy="50%"
                innerRadius={50} outerRadius={80} paddingAngle={3}
                label={({name,percent})=>`${name}: ${(percent*100).toFixed(0)}%`}
                labelLine={false}>
                <Cell fill="#F43F5E" /><Cell fill="#10B981" />
              </Pie>
              <Tooltip content={<TT />} />
              <Legend formatter={v=><span style={{color:'#E8EDF5',fontSize:12}}>{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Charts row 2 ── */}
      <div className="charts-grid">
        {/* Feature importance */}
        <div className="card">
          <div className="chart-title">Feature Importance (Coefficients)</div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={barData} layout="vertical">
              <XAxis type="number" tick={{ fill:'#5A7090', fontSize:10 }} />
              <YAxis dataKey="name" type="category" tick={{ fill:'#5A7090', fontSize:10 }} width={130} />
              <Tooltip content={<TT />} />
              <Bar dataKey="value" radius={[0,4,4,0]}>
                {barData.map((d,i)=>( <Cell key={i} fill={d.value>=0?'#10B981':'#F43F5E'} /> ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar */}
        <div className="card">
          <div className="chart-title">Feature Radar</div>
          <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(255,255,255,0.06)" />
              <PolarAngleAxis dataKey="feature" tick={{ fill:'#5A7090', fontSize:10 }} />
              <Radar name="Importance" dataKey="importance"
                stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.15} />
              <Tooltip content={<TT />} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── MongoDB live pie + confusion ── */}
      <div className="charts-grid">
        <div className="card">
          <div className="chart-title">Live MongoDB Predictions</div>
          {pieDb && db.total > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={pieDb} dataKey="value" cx="50%" cy="50%"
                  outerRadius={80} paddingAngle={3}
                  label={({name,value})=>`${name}: ${value}`}
                  labelLine={false}>
                  <Cell fill="#F43F5E" /><Cell fill="#10B981" />
                </Pie>
                <Tooltip content={<TT />} />
                <Legend formatter={v=><span style={{color:'#E8EDF5',fontSize:12}}>{v}</span>} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty-state">
              <div className="ei">🔍</div>
              <p>No predictions yet.<br/>Make a prediction to see live data.</p>
            </div>
          )}
        </div>

        {/* Confusion matrix */}
        <div className="card">
          <div className="chart-title">Confusion Matrix</div>
          {ml.confusion && (() => {
            const { tn, fp, fn, tp } = ml.confusion
            const cells = [
              { label:'True Negative', val:tn, desc:'Correctly Rejected', color:'var(--red)' },
              { label:'False Positive', val:fp, desc:'Wrongly Approved', color:'rgba(244,63,94,.4)' },
              { label:'False Negative', val:fn, desc:'Wrongly Rejected', color:'rgba(16,185,129,.4)' },
              { label:'True Positive', val:tp, desc:'Correctly Approved', color:'var(--green)' },
            ]
            return (
              <div>
                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:10, marginBottom:14 }}>
                  {cells.map((c,i)=>(
                    <div key={i} style={{ background:'var(--bg3)', borderRadius:10, padding:'14px', border:`1px solid ${c.color}33` }}>
                      <div style={{ fontFamily:'var(--mono)', fontSize:26, fontWeight:500, color:c.color }}>{c.val}</div>
                      <div style={{ fontSize:11, fontWeight:600, marginTop:4 }}>{c.label}</div>
                      <div style={{ fontSize:10, color:'var(--muted)', marginTop:2 }}>{c.desc}</div>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize:11, color:'var(--muted)', textAlign:'center' }}>
                  Predicted: Rejected ←→ Approved
                </div>
              </div>
            )
          })()}
        </div>
      </div>

      {/* ── Full coefficients table ── */}
      <div className="card" style={{ marginTop:0 }}>
        <div className="chart-title">All Feature Coefficients</div>
        <table className="tbl">
          <thead>
            <tr>
              <th>#</th><th>Feature</th><th>Coefficient</th><th>Impact Bar</th><th>Direction</th>
            </tr>
          </thead>
          <tbody>
            {coefData.map((d,i)=>(
              <tr key={i}>
                <td className="muted">{i+1}</td>
                <td style={{ fontWeight:500 }}>{d.name}</td>
                <td className="mono" style={{ color: d.value>=0?'var(--green)':'var(--red)' }}>
                  {d.value>=0?'+':''}{d.value.toFixed(4)}
                </td>
                <td style={{ width:200 }}>
                  <div className="factor-track">
                    <div className="factor-fill" style={{
                      width: (d.abs/coefData[0].abs*100)+'%',
                      background: d.value>=0?'var(--green)':'var(--red)',
                    }} />
                  </div>
                </td>
                <td>
                  <span className={`badge ${d.value>=0?'approved':'rejected'}`}>
                    {d.value>=0?'↑ Increases':'↓ Decreases'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
