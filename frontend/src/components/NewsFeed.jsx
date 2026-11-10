import { useEffect, useState, useCallback } from 'react'
import { api } from '../hooks/useApi'
import s from './NewsFeed.module.css'

/* relative "time ago" in Thai */
function timeAgo(iso) {
  if (!iso) return ''
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return ''
  const diff = Math.max(0, Date.now() - then)
  const m = Math.floor(diff / 60000)
  if (m < 1) return 'เมื่อสักครู่'
  if (m < 60) return `${m} นาทีที่แล้ว`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h} ชม.ที่แล้ว`
  const d = Math.floor(h / 24)
  if (d < 30) return `${d} วันที่แล้ว`
  return new Date(iso).toLocaleDateString('th-TH', { day: 'numeric', month: 'short' })
}

export default function NewsFeed() {
  const [items, setItems]     = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(false)
  const [source, setSource]   = useState('all')

  const load = useCallback(async (force = false) => {
    setLoading(true)
    setError(false)
    try {
      const res = await api.news(24, force)
      setItems(res?.items ?? [])
    } catch {
      setError(true)
      setItems([])
    }
    setLoading(false)
  }, [])

  useEffect(() => { load() }, [load])

  const visibleItems = items.filter(item => {
    if (source === 'investing') return item.source === 'Investing.com'
    if (source === 'yahoo') return item.source !== 'Investing.com'
    return true
  })

  return (
    <div className={s.wrap} data-testid="news-feed">
      <div className={s.header}>
        <h3>ข่าวล่าสุด</h3>
        <button
          className={s.refresh}
          onClick={() => load(true)}
          disabled={loading}
          data-testid="news-refresh-btn"
          aria-label="รีเฟรชข่าว"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
            className={loading ? s.spin : ''}>
            <path d="M21 12a9 9 0 1 1-2.64-6.36" />
            <polyline points="21 3 21 9 15 9" />
          </svg>
        </button>
      </div>

      <div className={s.sourceTabs} role="tablist" aria-label="แหล่งข่าว">
        {[
          ['all', 'ทั้งหมด'],
          ['investing', 'Investing.com'],
          ['yahoo', 'Yahoo Finance'],
        ].map(([value, label]) => (
          <button
            key={value}
            role="tab"
            aria-selected={source === value}
            className={source === value ? s.sourceTabActive : s.sourceTab}
            onClick={() => setSource(value)}
          >
            {label}
          </button>
        ))}
      </div>

      {loading ? (
        <ul className={s.list}>
          {Array.from({ length: 5 }).map((_, i) => (
            <li key={i} className={s.item}>
              <div className={`skeleton ${s.thumbSkel}`} />
              <div className={s.skelBody}>
                <div className={`skeleton ${s.lineSkel}`} style={{ width: '92%' }} />
                <div className={`skeleton ${s.lineSkel}`} style={{ width: '55%' }} />
              </div>
            </li>
          ))}
        </ul>
      ) : error ? (
        <div className={s.empty} data-testid="news-error">
          <p>โหลดข่าวไม่สำเร็จ ลองรีเฟรชอีกครั้ง</p>
        </div>
      ) : items.length === 0 || visibleItems.length === 0 ? (
        <div className={s.empty}>
          <p>{source === 'investing' ? 'ยังไม่มีข่าว Investing.com ในตอนนี้' : 'ยังไม่มีข่าวในตอนนี้'}</p>
        </div>
      ) : (
        <ul className={s.list} data-testid="news-list">
          {visibleItems.map((n, i) => (
            <li key={n.id ?? i} className={`${s.item} anim-up`} style={{ animationDelay: `${i * 30}ms` }}>
              <a
                href={n.url || '#'}
                target="_blank"
                rel="noopener noreferrer"
                className={s.link}
                data-testid={`news-item-${i}`}
              >
                {n.thumbnail ? (
                  <img src={n.thumbnail} alt="" className={s.thumb} loading="lazy" />
                ) : (
                  <div className={s.thumbFallback}>
                    <span>{n.source === 'Investing.com' ? 'INV' : n.ticker}</span>
                  </div>
                )}
                <div className={s.body}>
                  <p className={s.title}>{n.title}</p>
                  <div className={s.meta}>
                    <span className={s.ticker}>{n.ticker}</span>
                    {n.publisher && <span className={s.dot}>·</span>}
                    {n.publisher && <span className={s.publisher}>{n.publisher}</span>}
                    <span className={s.dot}>·</span>
                    <span className={s.time}>{timeAgo(n.published)}</span>
                  </div>
                </div>
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
