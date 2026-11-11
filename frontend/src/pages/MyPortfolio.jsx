import { useEffect, useState, useCallback, useRef } from 'react'
import { api } from '../hooks/useApi'
import { TICKER_GROUPS, ALL_PREDEFINED } from '../lib/tickers'
import PortfolioComparison from '../components/PortfolioComparison'
import s from './MyPortfolio.module.css'

const fmt = (n, d = 2) =>
  n != null ? Number(n).toLocaleString('th-TH', { minimumFractionDigits: d, maximumFractionDigits: d }) : '—'

const PORTFOLIO_ICONS = ['📊', '💰', '🎯', '🏠', '🚀', '🛡️', '📈', '💎']

export default function MyPortfolio() {
  // ── Multi-portfolio state ──
  const [portfolios, setPortfolios] = useState([])
  const [activePortfolioId, setActivePortfolioId] = useState(null)
  const [showPortfolioMenu, setShowPortfolioMenu] = useState(false)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [newPortfolioName, setNewPortfolioName] = useState('')
  const [newPortfolioIcon, setNewPortfolioIcon] = useState('📊')

  // ── Holdings state ──
  const [holdings, setHoldings] = useState([])
  const [loading, setLoading] = useState(true)
  const [livePrices, setLivePrices] = useState({})
  const [liveMeta, setLiveMeta] = useState(null)
  const [form, setForm] = useState({ ticker: '', shares: '', date: '' })
  const [adding, setAdding] = useState(false)
  const [error, setError] = useState('')
  const [customTickers, setCustomTickers] = useState([])
  const timerRef = useRef(null)

  // ── Currency state ──
  const [currency, setCurrency] = useState('THB') // 'THB' | 'USD'
  const fxRate = liveMeta?.fx_thb_per_usd ?? 35

  // ── Dividend state ──
  const [dividends, setDividends] = useState(null)
  const [divLoading, setDivLoading] = useState(false)

  const cur = (thbVal) => {
    if (thbVal == null) return null
    return currency === 'USD' ? thbVal / fxRate : thbVal
  }
  const curSym = currency === 'USD' ? '$' : '฿'

  // ── Load portfolios ──
  const loadPortfolios = useCallback(async () => {
    try {
      const data = await api.listPortfolios()
      const list = data?.portfolios || []
      setPortfolios(list)
      if (list.length && !activePortfolioId) {
        const savedId = localStorage.getItem('quantix.activePortfolio')
        setActivePortfolioId(list.some(p => p.id === savedId) ? savedId : list[0].id)
      }
    } catch (e) {
      console.warn('failed to load portfolios:', e.message)
    }
  }, [activePortfolioId])

  // ── Load holdings for active portfolio ──
  const loadHoldings = useCallback(async () => {
    if (!activePortfolioId) return
    try {
      const data = await api.getHoldings(activePortfolioId)
      setHoldings(data?.items || [])
    } catch (e) {
      console.warn('failed to load holdings:', e.message)
    } finally {
      setLoading(false)
    }
  }, [activePortfolioId])

  // ── Load custom tickers ──
  const loadCustomTickers = useCallback(async () => {
    try {
      const data = await api.getCustomTickers()
      setCustomTickers(data?.tickers || [])
    } catch (e) {
      console.warn('failed to load custom tickers:', e.message)
    }
  }, [])

  useEffect(() => {
    loadPortfolios()
    loadCustomTickers()
  }, [loadPortfolios, loadCustomTickers])

  useEffect(() => {
    if (activePortfolioId) localStorage.setItem('quantix.activePortfolio', activePortfolioId)
  }, [activePortfolioId])

  useEffect(() => {
    if (activePortfolioId) loadHoldings()
  }, [activePortfolioId, loadHoldings])

  // ── Load dividends ──
  const loadDividends = useCallback(async () => {
    if (!activePortfolioId || holdings.length === 0) {
      setDividends(null)
      return
    }
    setDivLoading(true)
    try {
      const data = await api.getDividends(activePortfolioId)
      setDividends(data)
    } catch (e) {
      console.warn('failed to load dividends:', e.message)
    } finally {
      setDivLoading(false)
    }
  }, [activePortfolioId, holdings.length])

  useEffect(() => {
    loadDividends()
  }, [loadDividends])

  const loadLive = useCallback(async () => {
    try {
      const live = await api.livePrices()
      if (live?.prices) {
        setLivePrices(live.prices)
        setLiveMeta(live)
      }
    } catch (e) {
      console.warn('live prices fetch failed:', e.message)
    }
  }, [])

  useEffect(() => {
    loadLive()
    timerRef.current = setInterval(loadLive, 60_000)
    return () => clearInterval(timerRef.current)
  }, [loadLive])

  // ── Portfolio actions ──
  const handleCreatePortfolio = async () => {
    if (!newPortfolioName.trim()) return
    try {
      const result = await api.createPortfolio({ name: newPortfolioName.trim(), icon: newPortfolioIcon })
      if (result) {
        setPortfolios(prev => [...prev, { ...result, item_count: 0 }])
        setActivePortfolioId(result.id)
        setHoldings([])
        setNewPortfolioName('')
        setShowCreateForm(false)
      }
    } catch (e) {
      setError(e.message)
    }
  }

  const handleDeletePortfolio = async (id) => {
    if (portfolios.length <= 1) return
    if (!confirm('ลบพอร์ตนี้และหุ้นทั้งหมดในพอร์ต?')) return
    try {
      await api.deletePortfolio(id)
      const updated = portfolios.filter(p => p.id !== id)
      setPortfolios(updated)
      if (activePortfolioId === id) {
        setActivePortfolioId(updated[0].id)
      }
    } catch (e) {
      setError(e.message)
    }
  }

  // ── Holding actions ──
  const handleAdd = async () => {
    setError('')
    const ticker = form.ticker.trim().toUpperCase()
    if (!ticker || !form.shares || !form.date) {
      setError('กรุณากรอกข้อมูลให้ครบ')
      return
    }
    if (parseFloat(form.shares) <= 0) {
      setError('จำนวนหุ้นต้องมากกว่า 0')
      return
    }
    setAdding(true)
    try {
      const result = await api.priceAt(ticker, form.date)
      if (!result || result.close_thb == null) {
        setError('ไม่สามารถดึงราคาในวันที่เลือกได้')
        return
      }
      const item = await api.addHoldingToPortfolio(activePortfolioId, {
        ticker,
        shares: parseFloat(form.shares),
        buyDate: result.date,
        buyPrice: result.close_thb,
      })
      if (item) {
        setHoldings(prev => [...prev, item])
        if (!ALL_PREDEFINED.includes(ticker)) {
          await api.addCustomTicker(ticker)
          setCustomTickers(prev => prev.includes(ticker) ? prev : [...prev, ticker])
        }
      }
      setForm({ ticker: '', shares: '', date: '' })
    } catch (e) {
      setError(e.message || 'เกิดข้อผิดพลาด')
    } finally {
      setAdding(false)
    }
  }

  const handleDelete = async (id) => {
    try {
      await api.deleteHoldingFromPortfolio(activePortfolioId, id)
      setHoldings(prev => prev.filter(h => h.id !== id))
    } catch (e) {
      console.warn('delete failed:', e.message)
    }
  }

  // ── CSV Export ──
  const handleExportCSV = () => {
    const headers = ['Ticker', 'Shares', 'Buy Date', `Buy Price (${currency})`, `Current Price (${currency})`, `Value (${currency})`, `P/L (${currency})`, 'P/L (%)']
    const rows = rowsData.map(r => [
      r.ticker,
      r.shares,
      r.buyDate,
      cur(r.buyPrice),
      cur(r.currentPrice) ?? '',
      cur(r.value) ?? '',
      cur(r.pl) ?? '',
      r.plPct != null ? r.plPct.toFixed(2) : '',
    ])
    const csv = [headers, ...rows]
      .map(row => row.map(cell => `"${cell}"`).join(','))
      .join('\n')
    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `portfolio-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const rowsData = holdings.map(h => {
    const currentPrice = livePrices[h.ticker] ?? null
    const cost   = h.buyPrice * h.shares
    const value  = currentPrice != null ? currentPrice * h.shares : null
    const pl     = value != null ? value - cost : null
    const plPct  = value != null && cost > 0 ? (pl / cost) * 100 : null
    return { ...h, currentPrice, cost, value, pl, plPct }
  })

  const totalCost  = rowsData.reduce((sum, r) => sum + r.cost, 0)
  const totalValue = rowsData.reduce((sum, r) => sum + (r.value ?? 0), 0)
  const totalPL    = totalValue - totalCost
  const totalPLPct = totalCost > 0 ? (totalPL / totalCost) * 100 : 0

  const activePortfolio = portfolios.find(p => p.id === activePortfolioId)

  return (
    <div className={s.page}>
      {/* Header */}
      <header className={s.header}>
        <div>
          <div className={s.titleRow}>
            <h1>My Portfolio</h1>
            {/* Portfolio selector */}
            {portfolios.length > 0 && (
              <div className={s.portfolioSelector}>
                <button
                  className={s.portfolioBtn}
                  onClick={() => setShowPortfolioMenu(v => !v)}
                >
                  <span className={s.portfolioIcon}>{activePortfolio?.icon || '📊'}</span>
                  <span className={s.portfolioName}>{activePortfolio?.name || 'เลือกพอร์ต'}</span>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="6 9 12 15 18 9"/>
                  </svg>
                </button>
                {showPortfolioMenu && (
                  <>
                    <div className={s.menuOverlay} onClick={() => setShowPortfolioMenu(false)} />
                    <div className={s.portfolioMenu}>
                      {portfolios.map(p => (
                        <div
                          key={p.id}
                          className={`${s.menuItem} ${p.id === activePortfolioId ? s.menuItemActive : ''}`}
                          onClick={() => {
                            setActivePortfolioId(p.id)
                            setShowPortfolioMenu(false)
                          }}
                        >
                          <span className={s.menuIcon}>{p.icon}</span>
                          <span className={s.menuText}>{p.name}</span>
                          <span className={s.menuCount}>{p.item_count}</span>
                          {portfolios.length > 1 && (
                            <button
                              className={s.menuDelete}
                              onClick={(e) => { e.stopPropagation(); handleDeletePortfolio(p.id) }}
                              title="ลบพอร์ต"
                            >✕</button>
                          )}
                        </div>
                      ))}
                      <div className={s.menuDivider} />
                      <div className={s.menuItem} onClick={() => { setShowCreateForm(true); setShowPortfolioMenu(false) }}>
                        <span className={s.menuIcon}>＋</span>
                        <span className={s.menuText}>สร้างพอร์ตใหม่</span>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
          <p className={s.sub}>
            ติดตามพอร์ต — เทียบราคาซื้อกับราคาปัจจุบัน
            {liveMeta?.date && <> · <time>{liveMeta.date}</time></>}
            {liveMeta && (
              <span className={s.liveBadge}>
                <span className={`pulse-dot ${liveMeta.status === 'live' ? 'green' : 'red'}`} />
                {liveMeta.status === 'live' ? 'Live' : 'Stale'}
              </span>
            )}
          </p>
        </div>
        <div className={s.headerActions}>
          {/* Currency toggle */}
          <div className={s.currencyToggle}>
            <button
              className={`${s.curBtn} ${currency === 'THB' ? s.curActive : ''}`}
              onClick={() => setCurrency('THB')}
            >฿ THB</button>
            <button
              className={`${s.curBtn} ${currency === 'USD' ? s.curActive : ''}`}
              onClick={() => setCurrency('USD')}
            >$ USD</button>
          </div>
          {holdings.length > 0 && (
            <button className="btn btn-ghost" onClick={handleExportCSV}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="7 10 12 15 17 10"/>
                <line x1="12" y1="15" x2="12" y2="3"/>
              </svg>
              Export CSV
            </button>
          )}
        </div>
      </header>

      {/* Create portfolio form */}
      {showCreateForm && (
        <div className={`card ${s.createForm}`}>
          <h3>สร้างพอร์ตใหม่</h3>
          <div className={s.createRow}>
            <div className={s.iconPicker}>
              {PORTFOLIO_ICONS.map(ic => (
                <button
                  key={ic}
                  className={`${s.iconBtn} ${newPortfolioIcon === ic ? s.iconActive : ''}`}
                  onClick={() => setNewPortfolioIcon(ic)}
                >{ic}</button>
              ))}
            </div>
            <input
              className={s.input}
              type="text"
              placeholder="ชื่อพอร์ต (เช่น พอร์ตเกษียณ, พอร์ตเก็งกำไร)"
              value={newPortfolioName}
              onChange={e => setNewPortfolioName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleCreatePortfolio()}
            />
            <button className="btn btn-primary" onClick={handleCreatePortfolio}>สร้าง</button>
            <button className="btn btn-ghost" onClick={() => setShowCreateForm(false)}>ยกเลิก</button>
          </div>
        </div>
      )}

      {/* Summary */}
      {holdings.length > 0 && (
        <div className={s.summary}>
          <div className={`card ${s.sumCard}`}>
            <span className={s.sumLabel}>ต้นทุนรวม</span>
            <span className={s.sumValue}>{curSym}{fmt(cur(totalCost))}</span>
          </div>
          <div className={`card ${s.sumCard}`}>
            <span className={s.sumLabel}>มูลค่าปัจจุบัน</span>
            <span className={s.sumValue}>{curSym}{fmt(cur(totalValue))}</span>
          </div>
          <div className={`card ${s.sumCard} ${totalPL >= 0 ? s.profit : s.loss}`}>
            <span className={s.sumLabel}>กำไร/ขาดทุน</span>
            <span className={s.sumValue}>
              {totalPL >= 0 ? '+' : '-'}{curSym}{fmt(Math.abs(cur(totalPL)))}
            </span>
          </div>
          <div className={`card ${s.sumCard} ${totalPL >= 0 ? s.profit : s.loss}`}>
            <span className={s.sumLabel}>ผลตอบแทน</span>
            <span className={s.sumValue}>
              {totalPLPct >= 0 ? '+' : ''}{fmt(totalPLPct)}%
            </span>
          </div>
        </div>
      )}

      {/* Dividend summary */}
      {holdings.length > 0 && dividends && dividends.total_dividends > 0 && (
        <div className={`card ${s.divCard}`}>
          <h3 className={s.cardTitle}>เงินปันผล + ผลตอบแทนรวม (Total Return)</h3>
          <div className={s.divSummary}>
            <div className={s.divMetric}>
              <span className={s.divLabel}>ปันผลรวม</span>
              <span className={s.divValue}>฿{fmt(dividends.total_dividends)}</span>
            </div>
            <div className={s.divMetric}>
              <span className={s.divLabel}>กำไร/ขาดทุนจากราคา</span>
              <span className={`${s.divValue} ${dividends.capital_gain >= 0 ? s.profit : s.loss}`}>
                {dividends.capital_gain >= 0 ? '+' : '-'}฿{fmt(Math.abs(dividends.capital_gain))}
              </span>
            </div>
            <div className={s.divMetric}>
              <span className={s.divLabel}>ผลตอบแทนรวม</span>
              <span className={`${s.divValue} ${dividends.total_return_value >= 0 ? s.profit : s.loss}`}>
                {dividends.total_return_value >= 0 ? '+' : '-'}฿{fmt(Math.abs(dividends.total_return_value))}
              </span>
            </div>
            <div className={s.divMetric}>
              <span className={s.divLabel}>Total Return %</span>
              <span className={`${s.divValue} ${dividends.total_return_pct >= 0 ? s.profit : s.loss}`}>
                {dividends.total_return_pct >= 0 ? '+' : ''}{fmt(dividends.total_return_pct)}%
              </span>
            </div>
          </div>
          <div className={s.divBreakdown}>
            <div className={s.divBarRow}>
              <span className={s.divBarLabel}>ผลตอบแทนจากราคา</span>
              <div className={s.divBarBg}>
                <div className={s.divBarFill} style={{
                  width: `${Math.min(Math.abs(dividends.capital_return_pct), 100)}%`,
                  background: dividends.capital_return_pct >= 0 ? 'var(--green)' : 'var(--red)',
                }} />
              </div>
              <span className={s.divBarPct}>{fmt(dividends.capital_return_pct)}%</span>
            </div>
            <div className={s.divBarRow}>
              <span className={s.divBarLabel}>ผลตอบแทนจากปันผล</span>
              <div className={s.divBarBg}>
                <div className={s.divBarFill} style={{
                  width: `${Math.min(dividends.dividend_return_pct * 3, 100)}%`,
                  background: 'var(--accent)',
                }} />
              </div>
              <span className={s.divBarPct}>{fmt(dividends.dividend_return_pct)}%</span>
            </div>
          </div>
          {dividends.dividends.length > 0 && (
            <details className={s.divDetails}>
              <summary>ดูประวัติการจ่ายปันผล ({dividends.dividends.length} ครั้ง)</summary>
              <div className={s.tableWrap}>
                <table className={s.table}>
                  <thead>
                    <tr>
                      <th>หุ้น</th>
                      <th>วันที่</th>
                      <th className={s.right}>ปันผล/หุ้น (฿)</th>
                      <th className={s.right}>จำนวนหุ้น</th>
                      <th className={s.right}>รวม (฿)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dividends.dividends.map((d, i) => (
                      <tr key={i}>
                        <td className={s.ticker}>{d.ticker}</td>
                        <td className={s.dateCell}>{d.date}</td>
                        <td className={s.right}>{fmt(d.per_share_thb, 4)}</td>
                        <td className={s.right}>{fmt(d.shares, 0)}</td>
                        <td className={s.right}>฿{fmt(d.total)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </div>
      )}
      {divLoading && holdings.length > 0 && (
        <div className={`card ${s.divLoading}`}>กำลังคำนวณเงินปันผล…</div>
      )}

      {/* Add form */}
      <div className={`card ${s.formCard}`}>
        <h3>เพิ่มหุ้นที่ซื้อ</h3>
        <div className={s.form}>
          <input
            className={`${s.select} ${s.tickerInput}`}
            type="text"
            list="ticker-list"
            placeholder="เลือกหรือพิมพ์ ticker…"
            value={form.ticker}
            onChange={e => setForm(f => ({ ...f, ticker: e.target.value.toUpperCase() }))}
          />
          <datalist id="ticker-list">
            {TICKER_GROUPS.flatMap(g => g.items).map(t => <option key={t} value={t} />)}
            {customTickers.map(t => <option key={t} value={t} />)}
          </datalist>
          <input
            className={s.input}
            type="number"
            placeholder="จำนวนหุ้น"
            value={form.shares}
            onChange={e => setForm(f => ({ ...f, shares: e.target.value }))}
            min="0"
            step="any"
          />
          <input
            className={s.input}
            type="date"
            value={form.date}
            max={new Date().toISOString().split('T')[0]}
            onChange={e => setForm(f => ({ ...f, date: e.target.value }))}
          />
          <button
            className="btn btn-primary"
            onClick={handleAdd}
            disabled={adding || !activePortfolioId}
          >
            {adding ? <span className="spinner" /> : 'เพิ่ม'}
          </button>
        </div>
        {error && <p className={s.error}>{error}</p>}
      </div>

      {/* Portfolio vs MPT comparison */}
      {holdings.length > 0 && (
        <PortfolioComparison holdings={holdings} livePrices={livePrices} />
      )}

      {/* Holdings table */}
      {loading ? (
        <div className={`card ${s.empty}`}>กำลังโหลด…</div>
      ) : holdings.length > 0 ? (
        <div className={`card ${s.tableCard}`}>
          <div className={s.tableWrap}>
            <table className={s.table}>
              <thead>
                <tr>
                  <th>หุ้น</th>
                  <th className={s.right}>จำนวน</th>
                  <th>วันที่ซื้อ</th>
                  <th className={s.right}>ราคาซื้อ ({curSym})</th>
                  <th className={s.right}>ราคาล่าสุด ({curSym})</th>
                  <th className={s.right}>มูลค่า ({curSym})</th>
                  <th className={s.right}>กำไร/ขาดทุน</th>
                  <th className={s.right}>%</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {rowsData.map(r => (
                  <tr key={r.id}>
                    <td className={s.ticker}>{r.ticker}</td>
                    <td className={s.right}>{fmt(r.shares, 0)}</td>
                    <td className={s.dateCell}>{r.buyDate}</td>
                    <td className={s.right}>{curSym}{fmt(cur(r.buyPrice))}</td>
                    <td className={s.right}>
                      {r.currentPrice != null ? `${curSym}${fmt(cur(r.currentPrice))}` : '—'}
                    </td>
                    <td className={s.right}>
                      {r.value != null ? `${curSym}${fmt(cur(r.value))}` : '—'}
                    </td>
                    <td className={`${s.right} ${r.pl != null ? (r.pl >= 0 ? s.profit : s.loss) : ''}`}>
                      {r.pl != null ? `${r.pl >= 0 ? '+' : '-'}${curSym}${fmt(Math.abs(cur(r.pl)))}` : '—'}
                    </td>
                    <td className={`${s.right} ${r.plPct != null ? (r.plPct >= 0 ? s.profit : s.loss) : ''}`}>
                      {r.plPct != null ? `${r.plPct >= 0 ? '+' : ''}${fmt(r.plPct)}%` : '—'}
                    </td>
                    <td>
                      <button className={s.delBtn} onClick={() => handleDelete(r.id)} title="ลบ">✕</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className={`card ${s.empty}`}>
          ยังไม่มีหุ้นในพอร์ต — เพิ่มหุ้นที่ซื้อด้านบนเพื่อเริ่มติดตามกำไร/ขาดทุน
        </div>
      )}
    </div>
  )
}
