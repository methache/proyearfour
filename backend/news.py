"""
Aggregate Yahoo Finance and Investing.com RSS headlines with TTL caching.

Exposes:
  * GET /api/news  → latest deduplicated portfolio and Investing.com headlines.

Fetches per-ticker news concurrently and caches the merged result for
`CACHE_TTL` seconds to keep latency low.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree

import httpx
import yfinance as yf
from fastapi import FastAPI

from yh import ALL_TICKERS

log = logging.getLogger("news")

CACHE_TTL = 600          # seconds — merged news snapshot reuse window
PER_TICKER = 4           # max headlines pulled per ticker before merge
MAX_ITEMS = 40           # cap on merged list
INVESTING_RSS_URL = os.getenv(
    "INVESTING_RSS_URL", "https://th.investing.com/rss/news_25.rss"
)

_cache: list[dict[str, Any]] | None = None
_cache_ts: float = 0.0
_lock = asyncio.Lock()


def _fetch_one(ticker: str) -> list[dict[str, Any]]:
    # Yahoo's older Ticker.news endpoint currently returns no stories for all
    # symbols; the Search endpoint remains available in yfinance 1.7.x.
    try:
        items = yf.Search(ticker, max_results=0, news_count=PER_TICKER).news or []
    except Exception as e:  # noqa: BLE001
        log.debug("Yahoo Finance search news failed for %s: %s", ticker, e)
        return []

    return _normalize_yahoo_news(items, ticker)[:PER_TICKER]


def _normalize_yahoo_news(items: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        content = item.get("content") or item
        title = content.get("title")
        if not title:
            continue
        url_value = content.get("clickThroughUrl") or content.get("canonicalUrl") or content.get("link")
        url = url_value.get("url") if isinstance(url_value, dict) else url_value
        thumbnail = content.get("thumbnail") or {}
        resolutions = thumbnail.get("resolutions") or [] if isinstance(thumbnail, dict) else []
        if resolutions:
            small = [r for r in resolutions if r.get("tag") and r["tag"] != "original"]
            image = (small[0] if small else resolutions[0]).get("url")
        else:
            image = thumbnail if isinstance(thumbnail, str) else None

        published = content.get("pubDate") or content.get("displayTime")
        if not published and content.get("providerPublishTime"):
            try:
                published = datetime.fromtimestamp(
                    float(content["providerPublishTime"]), tz=timezone.utc
                ).isoformat()
            except (TypeError, ValueError, OverflowError):
                published = None
        provider = content.get("provider") or {}
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)
        publisher = publisher or content.get("publisher", "")
        out.append({
            "id": content.get("id") or content.get("uuid") or item.get("id") or item.get("uuid"),
            "ticker": ticker,
            "source": "Yahoo Finance",
            "title": title,
            "summary": (content.get("summary") or content.get("description") or "").strip(),
            "publisher": publisher,
            "url": url,
            "thumbnail": image,
            "published": published,
        })
    return out


def _fetch_investing_news() -> list[dict[str, Any]]:
    """Read Investing.com's official Thai stock-market RSS feed."""
    try:
        response = httpx.get(
            INVESTING_RSS_URL,
            headers={"User-Agent": "QuantixNews/1.0 (RSS reader)"},
            timeout=8,
            follow_redirects=True,
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
    except Exception as e:  # noqa: BLE001
        log.warning("Investing.com RSS fetch failed: %s", e)
        return []

    out: list[dict[str, Any]] = []
    for item in root.findall(".//item")[:MAX_ITEMS]:
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        if not title or not url:
            continue

        guid = (item.findtext("guid") or url or title).strip()
        published = (item.findtext("pubDate") or "").strip()
        if published:
            try:
                parsed = parsedate_to_datetime(published)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                published = parsed.astimezone(timezone.utc).isoformat()
            except (TypeError, ValueError, OverflowError):
                published = ""

        out.append({
            "id": f"investing-{hashlib.sha256(guid.encode('utf-8')).hexdigest()[:24]}",
            "ticker": "ข่าวตลาดหุ้น",
            "source": "Investing.com",
            "title": title,
            "summary": "",
            "publisher": "Investing.com",
            "url": url,
            "thumbnail": None,
            "published": published,
        })
    return out


def _parse_ts(item: dict[str, Any]) -> datetime:
    p = item.get("published") or ""
    try:
        return datetime.fromisoformat(p.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return datetime.min.replace(tzinfo=timezone.utc)


def _download_all() -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for chunk in ex.map(_fetch_one, ALL_TICKERS):
            merged.extend(chunk)

    # A broad Yahoo search keeps the source tab populated if symbol-specific
    # searches are sparse or Yahoo returns no ticker results in this region.
    yahoo_count = sum(item.get("source") == "Yahoo Finance" for item in merged)
    if yahoo_count < 8:
        try:
            broad_news = yf.Search("stock market", max_results=0, news_count=20).news or []
            merged.extend(_normalize_yahoo_news(broad_news, "ตลาดหุ้น"))
        except Exception as e:  # noqa: BLE001
            log.warning("Yahoo Finance broad news search failed: %s", e)

    merged.extend(_fetch_investing_news())

    # dedupe by id (fallback title) — same story surfaces on multiple tickers
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for n in merged:
        key = str(n.get("id") or n.get("title"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(n)

    uniq.sort(key=_parse_ts, reverse=True)
    # Keep both sources available to the UI's source filters.
    investing = [item for item in uniq if item.get("source") == "Investing.com"][:20]
    other = [item for item in uniq if item.get("source") != "Investing.com"][:20]
    return sorted([*investing, *other], key=_parse_ts, reverse=True)[:MAX_ITEMS]


def _limit_news_by_source(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    investing = [item for item in items if item.get("source") == "Investing.com"]
    other = [item for item in items if item.get("source") != "Investing.com"]
    if not investing or not other or limit <= 1:
        return items[:limit]

    investing_count = min(len(investing), max(1, limit // 3))
    selected = [*investing[:investing_count], *other[:max(0, limit - investing_count)]]
    if len(selected) < limit:
        selected.extend(investing[investing_count:investing_count + limit - len(selected)])
    return sorted(selected, key=_parse_ts, reverse=True)


async def get_news(force: bool = False) -> list[dict[str, Any]]:
    global _cache, _cache_ts
    now = time.time()
    if not force and _cache and (now - _cache_ts) < CACHE_TTL:
        return _cache

    async with _lock:
        if not force and _cache and (time.time() - _cache_ts) < CACHE_TTL:
            return _cache
        try:
            data = await asyncio.get_event_loop().run_in_executor(None, _download_all)
            _cache = data
            _cache_ts = time.time()
            return data
        except Exception as e:  # noqa: BLE001
            log.warning("news download failed: %s", e)
            return _cache or []


async def get_news_snapshot(force: bool = False) -> dict[str, Any]:
    """Return cached headlines with the actual fetch time and freshness."""
    items = await get_news(force=force)
    age = time.time() - _cache_ts if _cache_ts else None
    if not _cache_ts:
        status = "unavailable"
    elif age is not None and age > CACHE_TTL:
        status = "stale"
    else:
        status = "live"
    fetched_at = (
        datetime.fromtimestamp(_cache_ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
        if _cache_ts else None
    )
    return {"items": items, "fetched_at": fetched_at, "status": status}


def register(app: FastAPI) -> None:
    @app.get("/api/news")
    async def api_news(limit: int = 30, force: bool = False):
        snapshot = await get_news_snapshot(force=force)
        items = _limit_news_by_source(snapshot["items"], limit)
        return {
            "count": len(items),
            "items": items,
            "fetched_at": snapshot["fetched_at"],
            "status": snapshot["status"],
        }

    @app.on_event("startup")
    async def _warm_news() -> None:
        asyncio.create_task(get_news(force=True))
