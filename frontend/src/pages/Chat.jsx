import { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { api }      from '../hooks/useApi'
import { useChart } from '../context/ChartContext'
import s            from './Chat.module.css'

/* current time label (module-scope → stable, no hook dep churn) */
const now = () =>
  new Date().toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit' })

/* stable unique id for message keys */
const uid = () =>
  (typeof crypto !== 'undefined' && crypto.randomUUID)
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`

/* ── Suggestions ────────────────────────────── */
const SUGGESTIONS = [
  { icon: '💼', text: 'จัดพอร์ตให้หน่อย Sharpe ดีที่สุด' },
  { icon: '🧭', text: 'ตลาดตอนนี้ Bull หรือ Bear?' },
  { icon: '💰', text: 'ดูราคาหุ้นล่าสุดทุกตัว' },
  { icon: '🔄', text: 'อัพเดทข้อมูลราคาล่าสุด' },
  { icon: '📉', text: 'พอร์ต Volatility ต่ำที่สุดคืออะไร?' },
  { icon: '📋', text: 'สรุปพอร์ตปัจจุบันให้หน่อย' },
]

/* ── Tool labels ────────────────────────────── */
const TOOL_LABELS = {
  optimize_portfolio:     'MPT Optimizer',
  classify_market_regime: 'Market Regime',
  get_portfolio_summary:  'Portfolio Summary',
  get_regime_summary:     'Regime Summary',
  get_latest_prices:      'Latest Prices',
  run_data_pipeline:      'Data Pipeline',
  get_backtest_results:   'Backtest Results',
  get_mpt_analysis:       'MPT Analysis',
  get_regime_analysis:    'Regime Analysis',
  get_live_market_data:   'Live Market Data',
  get_market_news:        'Market News',
}

/* ── Inline bold renderer ────────────────────── */
function RichText({ text }) {
  if (!text) return null
  return text.split('\n').map((line, i, arr) => {
    const parts = line.split(/\*\*(.*?)\*\*/g)
    return (
      <span key={i}>
        {parts.map((p, j) => j % 2 === 1 ? <strong key={j}>{p}</strong> : p)}
        {i < arr.length - 1 && <br />}
      </span>
    )
  })
}

/* ── Tool result card ────────────────────────── */
function ToolCard({ tool, result }) {
  const [open, setOpen] = useState(false)
  const ok = result?.status !== 'error'
  return (
    <div className={s.toolCard}>
      <button className={s.toolHead} onClick={() => setOpen(o => !o)}>
        <span className={`${s.toolDot} ${ok ? s.toolOk : s.toolErr}`} />
        <span className={s.toolName}>{TOOL_LABELS[tool] ?? tool}</span>
        <span className={s.toolChev}>{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <pre className={s.toolPre}>{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  )
}

/* ── Message bubble ──────────────────────────── */
function Bubble({ msg, onRetry }) {
  const [copied, setCopied] = useState(false)
  const isUser = msg.role === 'user'
  const copyMessage = async () => {
    try {
      const text = msg.content || ''
      try {
        if (!navigator.clipboard?.writeText) throw new Error('Clipboard API unavailable')
        await navigator.clipboard.writeText(text)
      } catch {
        const temporary = document.createElement('textarea')
        temporary.value = text
        temporary.style.position = 'fixed'
        temporary.style.opacity = '0'
        document.body.appendChild(temporary)
        temporary.select()
        const copied = document.execCommand('copy')
        temporary.remove()
        if (!copied) throw new Error('Copy failed')
      }
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1400)
    } catch {
      setCopied(false)
    }
  }
  return (
    <div className={`${s.bRow} ${isUser ? s.bRowUser : ''}`}>
      <div className={`${s.avatar} ${isUser ? s.avatarUser : s.avatarBot}`}>
        {isUser ? 'U' : 'AI'}
      </div>
      <div className={s.bContent}>
        <div className={s.bMeta}>
          <span className={s.bRole}>{isUser ? 'You' : 'Quant AI'}</span>
          {msg.time && <span className={s.bTime}>{msg.time}</span>}
        </div>
        <div className={`${s.bubble} ${isUser ? s.bUser : s.bBot}`}>
          <RichText text={msg.content} />
        </div>
        <div className={`${s.messageActions} ${isUser ? s.messageActionsUser : ''}`}>
          <button className={s.messageAction} onClick={copyMessage} aria-label="คัดลอกข้อความ" title="คัดลอกข้อความ">
            {copied ? 'คัดลอกแล้ว' : 'คัดลอก'}
          </button>
          {msg.cancelled && isUser && (
            <>
              <span className={s.cancelledLabel}>ยกเลิกแล้ว</span>
              <button className={`${s.messageAction} ${s.retryAction}`} onClick={() => onRetry?.(msg)}>
                ↻ ส่งอีกครั้ง
              </button>
            </>
          )}
        </div>
        {msg.toolResults?.length > 0 && (
          <div className={s.tools}>
            {msg.toolResults.map((tr, i) => (
              <ToolCard key={`${tr.tool}-${i}`} tool={tr.tool} result={tr.result} />
            ))}
          </div>
        )}
        {msg.chartUpdated && (
          <div className={s.chartPing}>
            <span className={s.pingDot} />
            กราฟอัพเดทแล้ว
          </div>
        )}
      </div>
    </div>
  )
}

/* ── Typing dots ─────────────────────────────── */
function Typing() {
  return (
    <div className={s.bRow}>
      <div className={`${s.avatar} ${s.avatarBot}`}>AI</div>
      <div className={s.bContent}>
        <div className={s.bMeta}><span className={s.bRole}>Quant AI</span></div>
        <div className={`${s.bubble} ${s.bBot} ${s.bTyping}`}>
          <span /><span /><span />
        </div>
      </div>
    </div>
  )
}

/* ── Main Chat ───────────────────────────────── */
export default function Chat({ embedded = false }) {
  const { applyToolResults } = useChart()
  const navigate = useNavigate()

  const greeting = {
    id: 'greeting', role: 'bot',
    content: 'สวัสดีครับ ผมตอบคำถามเกี่ยวกับข้อมูลลงทุนใน Quantix โดยอ้างอิงผลจากระบบ เช่น MPT, สภาวะตลาด, ราคา และ backtest ถ้าไม่มีข้อมูลรองรับ ผมจะบอกตามตรงครับ',
    time: now(),
  }
  const [messages, setMessages] = useState([greeting])
  const [conversations, setConversations] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [historyFilter, setHistoryFilter] = useState('all')
  const [historyOpen, setHistoryOpen] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(true)
  const [editingId, setEditingId] = useState(null)
  const [editTitle, setEditTitle] = useState('')
  const [historyError, setHistoryError] = useState('')
  const [input, setInput] = useState('')
  const [busy,  setBusy]  = useState(false)

  const messagesRef = useRef(null)
  const textareaRef = useRef(null)
  const activeIdRef = useRef(null)
  const requestControllerRef = useRef(null)

  const setActive = (id) => {
    activeIdRef.current = id
    setActiveId(id)
    if (id) localStorage.setItem('quantix.activeChat', id)
    else localStorage.removeItem('quantix.activeChat')
  }

  const refreshConversations = useCallback(async () => {
    const data = await api.chatConversations()
    const items = data?.conversations ?? []
    setConversations(items)
    return items
  }, [])

  const selectConversation = useCallback(async (id) => {
    if (busy || !id) return
    setActive(id)
    setInput('')
    setHistoryOpen(false)
    setHistoryError('')
    setMessages([])
    setHistoryLoading(true)
    try {
      const data = await api.getChatConversation(id)
      if (activeIdRef.current !== id) return
      const loaded = (data.messages ?? []).map(m => ({
        id: m.id,
        role: m.role === 'user' ? 'user' : 'bot',
        content: m.content,
        toolResults: m.tool_results ?? [],
        time: m.created_at ? new Date(m.created_at).toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit' }) : '',
      }))
      setMessages(loaded.length ? loaded : [greeting])
    } catch (e) {
      if (activeIdRef.current === id) {
        setHistoryError(e.message)
        setActive(null)
        setMessages([greeting])
        setHistoryLoading(false)
      }
    } finally {
      if (activeIdRef.current === id) setHistoryLoading(false)
    }
  }, [busy])

  useEffect(() => {
    let cancelled = false
    api.chatConversations()
      .then(data => {
        if (cancelled) return
        const items = data?.conversations ?? []
        setConversations(items)
        const savedId = localStorage.getItem('quantix.activeChat')
        const preferred = items.find(c => c.id === savedId) ?? (embedded ? items[0] : null)
        if (preferred) selectConversation(preferred.id)
        else {
          setActive(null)
          setMessages([greeting])
          setHistoryLoading(false)
        }
      })
      .catch(e => {
        if (!cancelled) {
          setHistoryError(e.message)
          setHistoryLoading(false)
        }
      })
    return () => { cancelled = true }
  }, [])

  const createConversation = useCallback(async () => {
    const conversation = await api.createChatConversation()
    setConversations(prev => [conversation, ...prev.filter(c => c.id !== conversation.id)])
    setActive(conversation.id)
    setMessages([])
    setInput('')
    setHistoryOpen(false)
    return conversation.id
  }, [])

  const newConversation = async () => {
    if (busy) return
    setHistoryError('')
    try { await createConversation() }
    catch (e) { setHistoryError(e.message) }
  }

  const toggleBookmark = async (conversation, event) => {
    event.stopPropagation()
    try {
      const updated = await api.updateChatConversation(conversation.id, { bookmarked: !conversation.bookmarked })
      setConversations(prev => prev.map(c => c.id === updated.id ? updated : c))
    } catch (e) { setHistoryError(e.message) }
  }

  const saveTitle = async (id) => {
    const title = editTitle.trim()
    if (!title) return setEditingId(null)
    try {
      const updated = await api.updateChatConversation(id, { title })
      setConversations(prev => prev.map(c => c.id === id ? updated : c))
      setEditingId(null)
    } catch (e) { setHistoryError(e.message) }
  }

  const removeConversation = async (conversation, event) => {
    event.stopPropagation()
    if (!window.confirm(`ลบแชท “${conversation.title}” และข้อความทั้งหมดหรือไม่?`)) return
    try {
      await api.deleteChatConversation(conversation.id)
      setConversations(prev => prev.filter(c => c.id !== conversation.id))
      if (activeIdRef.current === conversation.id) {
        setActive(null)
        setMessages([greeting])
        setInput('')
      }
    } catch (e) { setHistoryError(e.message) }
  }

  useEffect(() => {
    const container = messagesRef.current
    if (container) container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' })
  }, [messages, busy])

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 130) + 'px'
  }, [input])

  const send = useCallback(async (text) => {
    const msg = text.trim()
    if (!msg || busy) return
    const request = {
      controller: new AbortController(),
      messageId: uid(),
      message: msg,
      messageShown: false,
    }
    requestControllerRef.current = request
    setInput('')
    setBusy(true)
    let conversationId = activeIdRef.current
    try {
      if (!conversationId) conversationId = await createConversation()
      if (request.controller.signal.aborted) {
        setInput(msg)
        return
      }
      setMessages(prev => [...prev, { id: request.messageId, role: 'user', content: msg, time: now() }])
      request.messageShown = true
      const data = await api.sendChatMessage(conversationId, msg, request.controller.signal)
      if (request.controller.signal.aborted) return
      const toolResults = data.tool_results ?? []
      if (toolResults.length > 0) applyToolResults(toolResults)
      const chartUpdated = toolResults.some(({ tool, result }) =>
        (tool === 'optimize_portfolio' && result?.objective === 'max_sharpe') ||
        ['classify_market_regime', 'get_portfolio_summary', 'get_regime_summary',
          'get_latest_prices', 'get_live_market_data', 'run_data_pipeline'].includes(tool)
      )
      if (activeIdRef.current === conversationId) {
        setMessages(prev => [...prev, {
          id: uid(), role: 'bot',
          content: data.reply?.trim() || 'เครื่องมือดึงข้อมูลแล้ว แต่ยังไม่ได้รับข้อความสรุปจากโมเดล ผลลัพธ์แสดงในการ์ดด้านล่างครับ',
          toolResults, chartUpdated, time: now(),
        }])
      }
      await refreshConversations()
    } catch (e) {
      if (!request.controller.signal.aborted && activeIdRef.current === conversationId) {
        setMessages(prev => [...prev, { id: uid(), role: 'bot', content: `❌ ${e.message}`, time: now() }])
      }
      refreshConversations().catch(() => {})
    } finally {
      if (requestControllerRef.current === request) requestControllerRef.current = null
      setBusy(false)
    }
  }, [busy, applyToolResults, createConversation, refreshConversations])

  const cancelReply = () => {
    const request = requestControllerRef.current
    if (!request) return
    request.controller.abort()
    if (request.messageShown) {
      setMessages(prev => prev.map(message => message.id === request.messageId
        ? { ...message, cancelled: true }
        : message))
    } else {
      setInput(request.message)
    }
  }

  const retryCancelledMessage = (message) => {
    if (busy) return
    setMessages(prev => prev.filter(item => item.id !== message.id))
    send(message.content)
  }

  const onKey = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) }
  }

  const showSuggestions = messages.length <= 1 && !busy
  const visibleConversations = conversations.filter(c => historyFilter === 'all' || c.bookmarked)

  return (
    <div className={`${s.page} ${embedded ? s.embedded : s.pageWithHistory}`}>

      {!embedded && (
        <aside className={`${s.historySidebar} ${historyOpen ? s.historySidebarOpen : ''}`}>
          <div className={s.historyTop}>
            <strong>ประวัติแชท</strong>
            <button className={s.newChatButton} onClick={newConversation} disabled={busy}>＋ แชทใหม่</button>
          </div>
          <div className={s.historyFilters}>
            <button className={historyFilter === 'all' ? s.filterActive : ''} onClick={() => setHistoryFilter('all')}>ทั้งหมด</button>
            <button className={historyFilter === 'bookmarked' ? s.filterActive : ''} onClick={() => setHistoryFilter('bookmarked')}>★ บันทึกไว้</button>
          </div>
          {historyError && <div className={s.historyError}>{historyError}</div>}
          <div className={s.conversationList}>
            {historyLoading && <div className={s.historyEmpty}>กำลังโหลด…</div>}
            {!historyLoading && visibleConversations.length === 0 && (
              <div className={s.historyEmpty}>{historyFilter === 'all' ? 'ยังไม่มีประวัติแชท' : 'ยังไม่มีแชทที่บันทึกไว้'}</div>
            )}
            {visibleConversations.map(c => (
              <div key={c.id} className={`${s.conversationRow} ${activeId === c.id ? s.conversationActive : ''}`}>
                {editingId === c.id ? (
                  <form className={s.renameForm} onSubmit={e => { e.preventDefault(); saveTitle(c.id) }}>
                    <input autoFocus value={editTitle} maxLength={80} onChange={e => setEditTitle(e.target.value)} />
                    <button type="submit" aria-label="บันทึกชื่อ">✓</button>
                    <button type="button" onClick={() => setEditingId(null)} aria-label="ยกเลิก">×</button>
                  </form>
                ) : (
                  <>
                    <button className={s.conversationSelect} onClick={() => selectConversation(c.id)} disabled={busy} title={c.title}>
                      <span className={s.conversationTitle}>{c.title}</span>
                      <span className={s.conversationDate}>{new Date(c.updated_at).toLocaleDateString('th-TH', { day: 'numeric', month: 'short' })}</span>
                    </button>
                    <div className={s.conversationActions}>
                      <button onClick={e => toggleBookmark(c, e)} title={c.bookmarked ? 'เลิกบันทึก' : 'บุ๊คมาร์ก'} aria-label="บุ๊คมาร์กแชท">{c.bookmarked ? '★' : '☆'}</button>
                      <button onClick={e => { e.stopPropagation(); setEditingId(c.id); setEditTitle(c.title) }} title="เปลี่ยนชื่อ" aria-label="เปลี่ยนชื่อแชท">✎</button>
                      <button onClick={e => removeConversation(c, e)} title="ลบแชท" aria-label="ลบแชท">×</button>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        </aside>
      )}

      <main className={s.chatMain}>

      {/* header */}
      <div className={s.header}>
        {embedded ? (
          <>
            <div className={s.headerLeft}>
              <span className={s.onlineDot} />
              <span className={s.title}>AI Chat</span>
            </div>
            <button className={s.historyLink} onClick={() => navigate('/chat')}>ประวัติ</button>
            <span className={s.model}>Gemini</span>
          </>
        ) : (
          <div className={s.headerLeft}>
            <button className={s.mobileHistoryButton} onClick={() => setHistoryOpen(v => !v)} aria-label="เปิดประวัติแชท">☰</button>
            <h1 className={s.title}>AI Chat</h1>
            <span className={s.sub}>{conversations.find(c => c.id === activeId)?.title ?? 'ประวัติแชทของคุณ'}</span>
          </div>
        )}
      </div>

      {historyError && embedded && <div className={s.historyError}>{historyError}</div>}
      {historyLoading && messages.length === 0 && <div className={s.historyLoading}>กำลังเปิดแชท…</div>}

      {/* suggestions */}
      {showSuggestions && (
        <div className={s.suggestions}>
          {SUGGESTIONS.map(({ icon, text }) => (
            <button key={text} className={s.chip} onClick={() => send(text)}>
              <span className={s.chipIcon}>{icon}</span>
              <span>{text}</span>
            </button>
          ))}
        </div>
      )}

      {/* messages */}
      <div className={s.messages} ref={messagesRef}>
        {messages.map((m) => <Bubble key={m.id} msg={m} onRetry={retryCancelledMessage} />)}
        {busy && <Typing />}
      </div>

      {/* input */}
      <div className={`${s.inputBar} ${busy ? s.inputDisabled : ''}`}>
        <textarea
          ref={textareaRef}
          className={s.input}
          rows={1}
          placeholder="ถามเกี่ยวกับพอร์ต, ตลาด หรือราคาหุ้น…"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKey}
          disabled={busy}
        />
        <button
          className={`${s.sendBtn} ${busy ? s.cancelBtn : ''}`}
          onClick={busy ? cancelReply : () => send(input)}
          disabled={!busy && !input.trim()}
          aria-label={busy ? 'ยกเลิกคำตอบ' : 'ส่ง'}
          title={busy ? 'ยกเลิกคำตอบ' : 'ส่งข้อความ'}
        >
          {busy
            ? <span className={s.stopIcon} />
            : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"/>
                <polygon points="22 2 15 22 11 13 2 9 22 2"/>
              </svg>
            )
          }
        </button>
      </div>
      </main>
      {historyOpen && !embedded && <button className={s.historyScrim} onClick={() => setHistoryOpen(false)} aria-label="ปิดประวัติแชท" />}
    </div>
  )
}
