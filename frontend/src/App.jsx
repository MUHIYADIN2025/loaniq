import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import Predict    from './pages/Predict'
import Dashboard  from './pages/Dashboard'
import History    from './pages/History'

const NAV = [
  { to: '/',          icon: '🔮', label: 'Predict'   },
  { to: '/dashboard', icon: '📊', label: 'Dashboard' },
  { to: '/history',   icon: '🗂️', label: 'History'   },
]

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sb-logo">
          <div className="sb-logo-title">LoanIQ</div>
          <div className="sb-logo-sub">ML Prediction System</div>
        </div>

        <nav className="sb-nav">
          {NAV.map(n => (
            <NavLink
              key={n.to} to={n.to} end={n.to === '/'}
              className={({ isActive }) => 'sb-link' + (isActive ? ' active' : '')}
            >
              <span className="icon">{n.icon}</span>
              {n.label}
            </NavLink>
          ))}

          <div className="sb-divider" />

          <div style={{ padding: '6px 12px', fontSize: 10, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '.8px' }}>
            System
          </div>
          <NavLink to="/dashboard" className="sb-link" style={{ fontSize: 13 }}>
            <span className="icon">⚙️</span>
            Model Stats
          </NavLink>
        </nav>

        <div className="sb-footer">
          <div>🍃 MongoDB</div>
          <div>🐍 Python ML</div>
          <div>⚡ Node + Express</div>
          <div>⚛️ React + Vite</div>
        </div>
      </aside>

      <main className="main">
        <Routes>
          <Route path="/"          element={<Predict />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/history"   element={<History />} />
        </Routes>
      </main>
    </div>
  )
}
