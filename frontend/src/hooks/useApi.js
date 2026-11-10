/**
 * useApi — fetch wrapper สำหรับ FastAPI backend
 * Vite proxy: /api/* และ /health → http://127.0.0.1:8001
 */
import { useState, useCallback } from 'react'

/* ── safe JSON parser ─────────────────────────────
 * ป้องกัน "Unexpected end of JSON input"
 * เมื่อ backend offline / proxy ไม่มี server ฟัง
 * ─────────────────────────────────────────────── */
async function safeJson(res) {
  const text = await res.text()
  if (!text || !text.trim()) {
    throw new Error(`Server returned empty response (HTTP ${res.status})`)
  }
  try {
    return JSON.parse(text)
  } catch {
    // ตัดข้อความยาวให้สั้นลงก่อน throw
    const preview = text.slice(0, 120).replace(/\n/g, ' ')
    throw new Error(`Invalid JSON from server: "${preview}"`)
  }
}

/* ── core fetch ───────────────────────────────── */
async function apiFetch(path, options = {}, timeoutMs = 15_000) {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  const externalSignal = options.signal
  const relayAbort = () => controller.abort(externalSignal?.reason)
  if (externalSignal?.aborted) relayAbort()
  else externalSignal?.addEventListener('abort', relayAbort, { once: true })

  try {
    const res = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      ...options,
      signal: controller.signal,
    })

    clearTimeout(timeout)

    if (!res.ok) {
      // พยายาม parse error message จาก body
      let detail = `HTTP ${res.status}`
      try {
        const err = await safeJson(res)
        detail = err.detail ?? err.message ?? detail
      } catch (e) {
        console.debug('could not parse error response body:', e.message)
      }
      throw new Error(detail)
    }

    return await safeJson(res)
  } catch (e) {
    if (externalSignal?.aborted) throw new Error('ยกเลิกคำขอแล้ว')
    if (e.name === 'AbortError') throw new Error('ระบบใช้เวลาตอบนานเกินกำหนด กรุณาลองอีกครั้ง')
    throw e
  } finally {
    clearTimeout(timeout)
    externalSignal?.removeEventListener('abort', relayAbort)
  }
}

/* ── React hook ───────────────────────────────── */
export function useApi() {
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)

  const call = useCallback(async (path, options = {}) => {
    setLoading(true)
    setError(null)
    try {
      const data = await apiFetch(path, options)
      return data
    } catch (e) {
      setError(e.message)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  return { call, loading, error }
}

/* ── Standalone API helpers ───────────────────── */
export const api = {
  health: () =>
    apiFetch('/health'),

  portfolioLatest: () =>
    apiFetch('/api/portfolio/latest'),

  regimeLatest: () =>
    apiFetch('/api/regime/latest'),

  prices: (n = 10) =>
    apiFetch(`/api/prices?n=${n}`),

  livePrices: () =>
    apiFetch('/api/prices/live'),

  runPipeline: (body = {}) =>
    apiFetch('/api/pipeline', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  runMpt: (body = {}) =>
    apiFetch('/api/mpt', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  runRegime: (body = {}) =>
    apiFetch('/api/regime', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  chat: (body) =>
    apiFetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify(body),
    }, 180_000),

  chatConversations: () =>
    apiFetch('/api/chat/conversations'),

  createChatConversation: () =>
    apiFetch('/api/chat/conversations', { method: 'POST' }),

  getChatConversation: (id) =>
    apiFetch(`/api/chat/conversations/${encodeURIComponent(id)}`),

  updateChatConversation: (id, body) =>
    apiFetch(`/api/chat/conversations/${encodeURIComponent(id)}`, {
      method: 'PATCH', body: JSON.stringify(body),
    }),

  deleteChatConversation: (id) =>
    apiFetch(`/api/chat/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' }),

  sendChatMessage: (id, message, signal) =>
    apiFetch(`/api/chat/conversations/${encodeURIComponent(id)}/messages`, {
      method: 'POST', body: JSON.stringify({ message }), signal,
    }, 180_000),

  returnsHistory: (period = '1y') =>
    apiFetch(`/api/returns-history?period=${period}`),

  news: (limit = 30, force = false) =>
    apiFetch(`/api/news?limit=${limit}${force ? '&force=true' : ''}`, {}, 30_000),

  analysisMpt: () =>
    apiFetch('/api/analysis/mpt'),

  analysisRegime: () =>
    apiFetch('/api/analysis/regime'),

  priceAt: (ticker, date) =>
    apiFetch(`/api/price/at?ticker=${encodeURIComponent(ticker)}&date=${encodeURIComponent(date)}`),

  // ── User data (portfolio, watchlist, tickers) ──
  getPortfolio: () =>
    apiFetch('/api/user/portfolio'),

  addHolding: (body) =>
    apiFetch('/api/user/portfolio', { method: 'POST', body: JSON.stringify(body) }),

  deleteHolding: (id) =>
    apiFetch(`/api/user/portfolio/${id}`, { method: 'DELETE' }),

  getWatchlist: () =>
    apiFetch('/api/user/watchlist'),

  addWatchlistItem: (body) =>
    apiFetch('/api/user/watchlist', { method: 'POST', body: JSON.stringify(body) }),

  deleteWatchlistItem: (id) =>
    apiFetch(`/api/user/watchlist/${id}`, { method: 'DELETE' }),

  checkAlerts: () =>
    apiFetch('/api/user/watchlist/check-alerts', { method: 'POST' }),

  getCustomTickers: () =>
    apiFetch('/api/user/tickers'),

  addCustomTicker: (ticker) =>
    apiFetch('/api/user/tickers', { method: 'POST', body: JSON.stringify({ ticker }) }),

  deleteCustomTicker: (ticker) =>
    apiFetch(`/api/user/tickers/${encodeURIComponent(ticker)}`, { method: 'DELETE' }),

  // ── Multi-portfolio ──
  listPortfolios: () =>
    apiFetch('/api/user/portfolios'),

  createPortfolio: (body) =>
    apiFetch('/api/user/portfolios', { method: 'POST', body: JSON.stringify(body) }),

  deletePortfolio: (id) =>
    apiFetch(`/api/user/portfolios/${id}`, { method: 'DELETE' }),

  updatePortfolio: (id, body) =>
    apiFetch(`/api/user/portfolios/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),

  getHoldings: (portfolioId) =>
    apiFetch(`/api/user/portfolios/${portfolioId}/holdings`),

  addHoldingToPortfolio: (portfolioId, body) =>
    apiFetch(`/api/user/portfolios/${portfolioId}/holdings`, { method: 'POST', body: JSON.stringify(body) }),

  deleteHoldingFromPortfolio: (portfolioId, itemId) =>
    apiFetch(`/api/user/portfolios/${portfolioId}/holdings/${itemId}`, { method: 'DELETE' }),

  // ── Dividends ──
  getDividends: (portfolioId) =>
    apiFetch(`/api/user/portfolios/${portfolioId}/dividends`),

  // ── Rebalancing ──
  rebalancingSuggestions: (holdings) =>
    apiFetch('/api/rebalancing/suggestions', { method: 'POST', body: JSON.stringify({ holdings }) }),

  // ── Backtesting ──
  backtest: (period = 'all') =>
    apiFetch(`/api/backtest?period=${period}`),

  // ── Trading terminal ──
  terminalSymbols: () =>
    apiFetch('/api/terminal/symbols'),

  terminalCandles: (symbol, tf = '1h') =>
    apiFetch(`/api/terminal/candles?symbol=${encodeURIComponent(symbol)}&tf=${encodeURIComponent(tf)}`),

  terminalQuotes: (symbols = []) =>
    apiFetch(`/api/terminal/quotes?symbols=${encodeURIComponent(symbols.join(','))}`),
}
