import { useState } from 'react'
import { api } from '../api'

const INIT = {
  gender:'Male', married:'Yes', dependents:'0', education:'Graduate',
  selfEmployed:'No', applicantIncome:'', coapplicantIncome:'',
  loanAmount:'', loanAmountTerm:'360', creditHistory:'1', propertyArea:'Urban',
}

const SEL = (k, label, opts) => ({ type:'select', key:k, label, opts })
const NUM = (k, label, ph)   => ({ type:'number', key:k, label, placeholder:ph })

const FIELDS = [
  SEL('gender',       'Gender',              ['Male','Female']),
  SEL('married',      'Marital Status',      ['Yes','No']),
  SEL('dependents',   'Dependents',          ['0','1','2','3+']),
  SEL('education',    'Education Level',     ['Graduate','Not Graduate']),
  SEL('selfEmployed', 'Self Employed',       ['No','Yes']),
  SEL('creditHistory','Credit History',      ['1','0']),
  SEL('propertyArea', 'Property Area',       ['Urban','Semiurban','Rural']),
  SEL('loanAmountTerm','Loan Term (months)', ['360','180','240','480','120','60']),
  NUM('applicantIncome',   'Applicant Income ($)',    'e.g. 5000'),
  NUM('coapplicantIncome', 'Co-applicant Income ($)', '0 if none'),
  { type:'number', key:'loanAmount', label:'Loan Amount (thousands $)', placeholder:'e.g. 150', span:true },
]

export default function Predict() {
  const [form,   setForm]   = useState(INIT)
  const [result, setResult] = useState(null)
  const [loading,setLoading]= useState(false)
  const [err,    setErr]    = useState('')

  const set = (k,v) => setForm(f => ({ ...f, [k]:v }))

  const submit = async () => {
    setErr(''); setResult(null)
    if (!form.applicantIncome || !form.loanAmount)
      return setErr('Please fill in Income and Loan Amount fields.')
    setLoading(true)
    try {
      const data = await api.predict(form)
      setResult(data)
    } catch(e) {
      setErr(e.message)
    } finally { setLoading(false) }
  }

  const maxCoef = result ? Math.max(...result.topFactors.map(f=>Math.abs(f.coefficient))) : 1

  return (
    <div>
      <div className="page-head">
        <h1>Loan <em>Prediction</em></h1>
        <p>Fill in applicant details — the ML model returns an instant decision.</p>
      </div>

      <div className="predict-grid">
        {/* ── Form ── */}
        <div className="card">
          <div style={{ display:'flex', alignItems:'center', gap:12, marginBottom:24 }}>
            <div style={{ fontSize:22 }}>📋</div>
            <div>
              <div style={{ fontWeight:600, fontSize:15 }}>Applicant Details</div>
              <div style={{ fontSize:12, color:'var(--muted)' }}>All fields are required</div>
            </div>
          </div>

          <div className="form-grid">
            {FIELDS.map(f => (
              <div key={f.key} className={`fg${f.span?' span2':''}`}>
                <label>{f.label}</label>
                {f.type === 'select' ? (
                  <select value={form[f.key]} onChange={e=>set(f.key,e.target.value)}>
                    {f.opts.map(o => (
                      <option key={o} value={o}>
                        {f.key==='creditHistory' ? (o==='1'?'Good (1)':'Bad (0)') : o}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number" placeholder={f.placeholder}
                    value={form[f.key]}
                    onChange={e=>set(f.key,e.target.value)}
                  />
                )}
              </div>
            ))}
          </div>

          {err && <div className="err">⚠️ {err}</div>}

          <button className="btn btn-primary" onClick={submit} disabled={loading}
            style={{ marginTop:20 }}>
            {loading
              ? <><span className="spinner" /> Predicting...</>
              : '🔮 Get Prediction'}
          </button>
        </div>

        {/* ── Result panel ── */}
        <div>
          {!result && (
            <div className="card" style={{ textAlign:'center', padding:'64px 28px' }}>
              <div style={{ fontSize:48, marginBottom:16, filter:'grayscale(1)', opacity:.4 }}>🏦</div>
              <div style={{ color:'var(--muted)', fontSize:14, lineHeight:1.8 }}>
                Submit the form<br/>to see the prediction
              </div>
            </div>
          )}

          {result && (
            <div className={`result ${result.status.toLowerCase()}`}>
              <div style={{ fontSize:11, color:'var(--muted)', textTransform:'uppercase', letterSpacing:'.8px', marginBottom:6 }}>
                Decision
              </div>
              <div className="result-verdict">
                {result.status === 'Approved' ? '✅ Approved' : '❌ Rejected'}
              </div>
              <div style={{ fontSize:13, color:'var(--muted2)', marginTop:4 }}>
                {result.status === 'Approved'
                  ? 'Applicant meets the criteria for loan approval.'
                  : 'Applicant does not meet the minimum criteria.'}
              </div>
              <div style={{ fontSize:11, color:'var(--muted)', marginTop:8 }}>
                Saved to MongoDB · ID: <span className="mono" style={{ fontSize:10 }}>{result._id}</span>
              </div>

              {/* Probability bars */}
              <div className="prob-row">
                <div className="prob-item">
                  <div className="prob-lbl">Approval %</div>
                  <div className="prob-track">
                    <div className="prob-fill g" style={{ width: result.probability.approved+'%' }} />
                  </div>
                  <div className="prob-num green">{result.probability.approved}%</div>
                </div>
                <div className="prob-item">
                  <div className="prob-lbl">Rejection %</div>
                  <div className="prob-track">
                    <div className="prob-fill r" style={{ width: result.probability.rejected+'%' }} />
                  </div>
                  <div className="prob-num red">{result.probability.rejected}%</div>
                </div>
              </div>

              {/* Top factors */}
              <div className="factors-head">Top Influencing Factors</div>
              {result.topFactors.map((f,i) => (
                <div key={i} className="factor">
                  <div className="factor-name">{f.feature}</div>
                  <div className="factor-track">
                    <div className="factor-fill" style={{
                      width: (Math.abs(f.coefficient)/maxCoef*100)+'%',
                      background: f.coefficient>0 ? 'var(--green)' : 'var(--red)',
                    }} />
                  </div>
                  <div className="factor-coef" style={{ color: f.coefficient>0?'var(--green)':'var(--red)' }}>
                    {f.coefficient>0?'+':''}{f.coefficient}
                  </div>
                </div>
              ))}

              <button className="btn btn-ghost" style={{ width:'100%', marginTop:20 }}
                onClick={() => { setResult(null); setForm(INIT) }}>
                🔄 New Prediction
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
