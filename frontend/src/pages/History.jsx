import { useEffect, useState } from 'react'
import { api } from '../api'

function fmt(iso) {
  return new Date(iso).toLocaleString('en-US', {
    month:'short', day:'numeric', year:'numeric',
    hour:'2-digit', minute:'2-digit'
  })
}

export default function History() {
  const [data,    setData]    = useState([])
  const [total,   setTotal]   = useState(0)
  const [pages,   setPages]   = useState(1)
  const [page,    setPage]    = useState(1)
  const [filter,  setFilter]  = useState('')
  const [loading, setLoading] = useState(true)
  const [err,     setErr]     = useState('')
  const [deleting,setDeleting]= useState(null)
  const [expanded,setExpanded]= useState(null)

  const load = async (p=1, f=filter) => {
    setLoading(true); setErr('')
    try {
      const res = await api.predictions(p, 10, f)
      setData(res.data); setTotal(res.total)
      setPages(res.pages); setPage(res.page)
    } catch(e) { setErr(e.message) }
    finally { setLoading(false) }
  }

  useEffect(()=>{ load(1, filter) }, [filter])

  const del = async id => {
    setDeleting(id)
    try { await api.deletePrediction(id); await load(page) }
    catch(e) { setErr(e.message) }
    finally { setDeleting(null) }
  }

  return (
    <div>
      <div className="page-head" style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-end' }}>
        <div>
          <h1>Prediction <em>History</em></h1>
          <p>{total} predictions saved in MongoDB.</p>
        </div>
        <div style={{ display:'flex', gap:8 }}>
          {['','Approved','Rejected'].map(f=>(
            <button key={f} className={`btn btn-ghost${filter===f?' active':''}`}
              onClick={()=>setFilter(f)}
              style={{ fontSize:12, padding:'7px 14px',
                ...(filter===f?{borderColor:'var(--amber)',color:'var(--amber)'}:{}) }}>
              {f||'All'}
            </button>
          ))}
          <button className="btn btn-ghost" style={{ fontSize:12, padding:'7px 14px' }}
            onClick={()=>load(page)}>
            🔄
          </button>
        </div>
      </div>

      {err && <div className="err">⚠️ {err}</div>}

      {loading ? (
        <div style={{ textAlign:'center', paddingTop:60 }}>
          <div className="spinner" style={{ width:28,height:28,borderWidth:3,borderTopColor:'var(--amber)' }} />
        </div>
      ) : data.length === 0 ? (
        <div className="card empty-state">
          <div className="ei">🗂️</div>
          <p>No predictions found.<br/>Go to <strong>Predict</strong> to create one.</p>
        </div>
      ) : (
        <>
          <div className="card" style={{ padding:0, overflow:'hidden' }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Status</th>
                  <th>Income</th>
                  <th>Loan Amount</th>
                  <th>Credit</th>
                  <th>Property</th>
                  <th>Approved %</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.map(row=>(
                  <>
                    <tr key={row._id} style={{ cursor:'pointer' }}
                      onClick={()=>setExpanded(expanded===row._id?null:row._id)}>
                      <td className="muted" style={{ fontSize:12 }}>{fmt(row.createdAt)}</td>
                      <td>
                        <span className={`badge ${row.status.toLowerCase()}`}>
                          {row.status==='Approved'?'✅':'❌'} {row.status}
                        </span>
                      </td>
                      <td className="mono">${Number(row.applicantIncome).toLocaleString()}</td>
                      <td className="mono">${row.loanAmount}k</td>
                      <td>
                        <span style={{ color: row.creditHistory===1?'var(--green)':'var(--red)', fontWeight:500 }}>
                          {row.creditHistory===1?'Good':'Bad'}
                        </span>
                      </td>
                      <td>{row.propertyArea}</td>
                      <td>
                        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                          <div style={{ flex:1, background:'var(--bg3)', borderRadius:99, height:5, width:60, overflow:'hidden' }}>
                            <div style={{ width:row.probability?.approved+'%', height:'100%', borderRadius:99,
                              background: row.status==='Approved'?'var(--green)':'var(--red)' }} />
                          </div>
                          <span className="mono" style={{ fontSize:12 }}>{row.probability?.approved}%</span>
                        </div>
                      </td>
                      <td onClick={e=>e.stopPropagation()}>
                        <button className="btn btn-danger"
                          onClick={()=>del(row._id)}
                          disabled={deleting===row._id}>
                          {deleting===row._id ? '...' : '🗑️'}
                        </button>
                      </td>
                    </tr>

                    {/* Expanded row */}
                    {expanded===row._id && (
                      <tr key={row._id+'_exp'}>
                        <td colSpan={8} style={{ background:'var(--bg3)', padding:'16px 20px' }}>
                          <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:16, marginBottom:14 }}>
                            {[
                              ['Gender',       row.gender],
                              ['Married',      row.married],
                              ['Dependents',   row.dependents],
                              ['Education',    row.education],
                              ['Self Employed',row.selfEmployed],
                              ['Co-Income',   `$${Number(row.coapplicantIncome).toLocaleString()}`],
                              ['Loan Term',   `${row.loanAmountTerm} mo.`],
                              ['MongoDB ID',   row._id.slice(-8)+'...'],
                            ].map(([k,v])=>(
                              <div key={k}>
                                <div style={{ fontSize:10, color:'var(--muted)', textTransform:'uppercase', letterSpacing:'.6px' }}>{k}</div>
                                <div style={{ fontSize:13, marginTop:3 }}>{v}</div>
                              </div>
                            ))}
                          </div>
                          {row.topFactors?.length > 0 && (
                            <div>
                              <div style={{ fontSize:10, color:'var(--muted)', textTransform:'uppercase', letterSpacing:'.6px', marginBottom:8 }}>
                                Top Factors
                              </div>
                              <div style={{ display:'flex', gap:8, flexWrap:'wrap' }}>
                                {row.topFactors.map((f,i)=>(
                                  <span key={i} style={{
                                    padding:'3px 10px', borderRadius:99, fontSize:11,
                                    background: f.coefficient>0?'rgba(16,185,129,.1)':'rgba(244,63,94,.1)',
                                    color: f.coefficient>0?'var(--green)':'var(--red)',
                                    border: `1px solid ${f.coefficient>0?'rgba(16,185,129,.2)':'rgba(244,63,94,.2)'}`,
                                  }}>
                                    {f.feature}: {f.coefficient>0?'+':''}{f.coefficient}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {pages > 1 && (
            <div className="pagination">
              <button className="pg-btn" disabled={page===1} onClick={()=>load(page-1)}>← Prev</button>
              {Array.from({length:pages},(_, i)=>(
                <button key={i} className={`pg-btn${page===i+1?' active':''}`}
                  onClick={()=>load(i+1)}>{i+1}</button>
              ))}
              <button className="pg-btn" disabled={page===pages} onClick={()=>load(page+1)}>Next →</button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
