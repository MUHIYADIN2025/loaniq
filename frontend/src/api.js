const BASE = '/api'

async function req(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } }
  if (body) opts.body = JSON.stringify(body)
  const res  = await fetch(BASE + path, opts)
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || 'Request failed')
  return data
}

export const api = {
  predict:        body => req('POST', '/predict', body),
  stats:          ()   => req('GET',  '/stats'),
  dbStats:        ()   => req('GET',  '/db-stats'),
  predictions:    (p=1,l=10,status='') =>
    req('GET', `/predictions?page=${p}&limit=${l}${status ? '&status='+status : ''}`),
  deletePrediction: id => req('DELETE', `/predictions/${id}`),
  retrain:        ()   => req('POST', '/retrain'),
  health:         ()   => req('GET',  '/health'),
}
