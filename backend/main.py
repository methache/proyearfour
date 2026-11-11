"""
FastAPI Backend — Quant AI Assistant
======================================
Endpoints:
  GET  /health                  — health check
  POST /api/pipeline            — รัน data pipeline (yh.py)
  POST /api/mpt                 — รัน MPT optimizer
  POST /api/regime              — รัน Market Regime classifier
  POST /api/chat                — LLM chat พร้อม Function Calling
  GET  /api/portfolio/latest    — ดึง optimal portfolio ล่าสุด
  GET  /api/regime/latest       — ดึงสภาวะตลาดล่าสุด
  GET  /api/prices              — ดึงราคาล่าสุด N แถว

LLM Function Calling:
  LLM (OpenAI-compatible) จะ parse คำถามผู้ใช้ แล้วเรียก tool ที่ถูกต้องโดยอัตโนมัติ
  รองรับ: OpenAI, Azure OpenAI, Ollama (openai-compatible)
"""

import os
import json
import asyncio
import re
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# โหลด .env ถ้ามี (ไม่ต้องติดตั้ง python-dotenv — อ่านไฟล์เอง)
for _env_path in (Path(__file__).parent.parent / ".env", Path(__file__).parent / ".env"):
    if _env_path.exists():
        for _line in _env_path.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip("\"'"))

# ─── optional LLM client (OpenAI SDK) ───
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

OUTPUT_DIR = Path("output")

app = FastAPI(
    title="Quant AI Assistant",
    description="MPT Portfolio Optimizer + Market Regime Classifier พร้อม LLM Function Calling",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────

class PipelineRequest(BaseModel):
    start_date: str = Field("2020-01-01", description="วันเริ่มต้น YYYY-MM-DD")
    end_date:   str = Field(
        default_factory=lambda: datetime.today().strftime("%Y-%m-%d"),
        description="วันสิ้นสุด"
    )

class MPTRequest(BaseModel):
    objective:     str   = Field("max_sharpe", description="max_sharpe | min_vol | max_return")
    top_n:         int | None = Field(None, description="เลือกเฉพาะ top N สินทรัพย์ (None = ทั้งหมด)")
    returns_csv:   str   = Field("output/daily_returns.csv")

class RegimeRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_type:  str  = Field("random_forest", description="random_forest | gradient_boosting")
    prices_csv:  str  = Field("output/prices_clean_thb.csv")
    save_model:  bool = Field(True)

class ChatRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    message:     str  = Field(...,  description="ข้อความจากผู้ใช้")
    history:     list[dict] = Field(default_factory=list, description="ประวัติการสนทนา")

class ChatResponse(BaseModel):
    reply:        str
    tool_calls:   list[dict] = []
    tool_results: list[dict] = []


# ─────────────────────────────────────────────
# LLM TOOL DEFINITIONS (Function Calling)
# ─────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_data_pipeline",
            "description": (
                "ดึงข้อมูลราคาหุ้น US, Thai, Bonds จาก Yahoo Finance "
                "แปลงเป็น THB และทำ data cleaning ก่อนการวิเคราะห์"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "วันเริ่มต้น รูปแบบ YYYY-MM-DD เช่น 2020-01-01",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "วันสิ้นสุด รูปแบบ YYYY-MM-DD",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_portfolio",
            "description": (
                "คำนวณน้ำหนักพอร์ตด้วย MPT จากผลตอบแทนย้อนหลัง เพื่อหา Sharpe สูงสุด, volatility ต่ำสุด หรือ return สูงสุด"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "enum": ["max_sharpe", "min_vol", "max_return"],
                        "description": "เป้าหมายการ optimize",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "จำนวนสินทรัพย์ที่ต้องการใช้ (เลือก top N จาก Sharpe สูงสุด)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_market_regime",
            "description": (
                "วิเคราะห์และจำแนกสภาวะตลาดปัจจุบัน (Bull/Bear/Neutral) "
                "ด้วย Random Forest โดยใช้ข้อมูลย้อนหลัง"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "enum": ["random_forest", "gradient_boosting"],
                        "description": "ประเภทโมเดล ML ที่ต้องการใช้",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_summary",
            "description": "อ่านพอร์ต MPT ที่บันทึกเป็นพอร์ตหลักอยู่ในขณะนี้เท่านั้น; ถ้าผู้ใช้ขอเป้าหมายอื่น เช่น Min Volatility หรือ Max Return ให้เรียก optimize_portfolio แทน",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_regime_summary",
            "description": "ดึงสภาวะตลาดล่าสุดพร้อม probability (Bull/Neutral/Bear)",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_prices",
            "description": "ดึงราคาสินทรัพย์ล่าสุด (หน่วย THB) พร้อม daily return",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_rows": {
                        "type": "integer",
                        "description": "จำนวนวันย้อนหลังที่ต้องการดู (default 5)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_backtest_results",
            "description": "ดึงผล backtest จริงเพื่อเปรียบเทียบ Buy & Hold, MPT และ Regime-Based",
            "parameters": {"type": "object", "properties": {
                "period": {"type": "string", "enum": ["1y", "3y", "all"]}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mpt_analysis",
            "description": "ดึงผลวิเคราะห์ MPT จริง เช่น optimal weights, return, volatility, Sharpe และ efficient frontier",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_regime_analysis",
            "description": "ดึง regime ล่าสุดและสถิติย้อนหลังจากข้อมูลจริง",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_market_data",
            "description": "ดึงราคาปิดล่าสุดและเปอร์เซ็นต์เปลี่ยนแปลงจาก Yahoo Finance API (ข้อมูลสดมี cache 60 วินาที) สำหรับหุ้นที่ระบบติดตาม",
            "parameters": {"type": "object", "properties": {
                "tickers": {"type": "array", "items": {"type": "string"},
                            "description": "สัญลักษณ์หุ้นที่ถาม เช่น AAPL หรือ PTT.BK"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_news",
            "description": "ดึงข่าวหุ้นล่าสุดจาก Yahoo Finance API สำหรับหุ้นที่ระบบติดตาม ใช้เมื่อผู้ใช้ถามข่าวหรือบริบทล่าสุด",
            "parameters": {"type": "object", "properties": {
                "ticker": {"type": "string", "description": "สัญลักษณ์หุ้นที่ต้องการกรองข่าว เช่น NVDA"}}, "required": []},
        },
    },
]


# ─────────────────────────────────────────────
# TOOL EXECUTION
# ─────────────────────────────────────────────

async def execute_tool(name: str, args: dict) -> dict:
    """dispatch tool calls → quant modules"""

    if name == "run_data_pipeline":
        from yh import run_pipeline
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_pipeline(
                start=args.get("start_date", "2020-01-01"),
                end=args.get("end_date", datetime.today().strftime("%Y-%m-%d")),
            )
        )
        return {
            "status": "success",
            "rows": len(result),
            "assets": len(result.columns),
            "message": f"ดึงข้อมูลสำเร็จ {len(result.columns)} สินทรัพย์ {len(result)} วัน",
        }

    elif name == "optimize_portfolio":
        from mpt import load_returns, optimize_portfolio

        def _optimize_for_chat():
            returns = load_returns()
            top_n = args.get("top_n")
            if top_n and top_n < len(returns.columns):
                from mpt import RISK_FREE_RATE, TRADING_DAYS
                individual_sharpe = (
                    (returns.mean() * TRADING_DAYS - RISK_FREE_RATE)
                    / (returns.std() * (TRADING_DAYS ** 0.5))
                )
                returns = returns[individual_sharpe.nlargest(top_n).index.tolist()]
            return optimize_portfolio(
                returns.mean(), returns.cov(),
                objective=args.get("objective", "max_sharpe"),
            )

        loop = asyncio.get_event_loop()
        opt = await loop.run_in_executor(None, _optimize_for_chat)
        top5 = sorted(opt["weights"].items(), key=lambda x: -x[1])[:5]
        return {
            "source": "output/daily_returns.csv",
            "as_of": str(load_returns().index[-1].date()),
            "status":                "success",
            "objective":             opt["objective"],
            "sharpe_ratio":          round(float(opt["sharpe_ratio"]), 4),
            "annualized_return":     round(float(opt["annualized_return"]), 4),
            "annualized_volatility": round(float(opt["annualized_volatility"]), 4),
            "top5_weights":          {t: round(w * 100, 4) for t, w in top5},
            "all_weights":           {t: round(w, 6) for t, w in opt["weights"].items()},
        }

    elif name == "classify_market_regime":
        from regime import run_regime
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_regime(model_type=args.get("model_type", "random_forest"))
        )
        curr = result["current_regime"]
        return {
            "status":       "success",
            "date":         curr["date"],
            "regime":       curr["regime_label"],
            "prob_bull":    round(float(curr["prob_bull"]), 4),
            "prob_neutral": round(float(curr["prob_neutral"]), 4),
            "prob_bear":    round(float(curr["prob_bear"]), 4),
            "cv_f1":        result["cv_f1_mean"],
        }

    elif name == "get_portfolio_summary":
        csv = OUTPUT_DIR / "optimal_portfolio.csv"
        if not csv.exists():
            return {"status": "error", "message": "ยังไม่มีข้อมูล กรุณารัน optimize_portfolio ก่อน"}
        df = pd.read_csv(csv)
        row = df.iloc[0].to_dict()
        # parse weights — CSV stored dict with numpy wrappers like {'AAPL': np.float64(0.1)}
        import re
        raw = row.get("weights", "") or ""
        weights = {t: float(v) for t, v in re.findall(r"'([^']+)':\s*(?:np\.float64\()?(-?[\d.]+)", raw)}
        top5 = sorted(weights.items(), key=lambda x: -x[1])[:5]
        return {
            "objective":             row.get("objective", "max_sharpe"),
            "sharpe_ratio":          round(float(row.get("sharpe_ratio", 0)), 4),
            "annualized_return":     round(float(row.get("annualized_return", 0)), 4),
            "annualized_volatility": round(float(row.get("annualized_volatility", 0)), 4),
            "top5_weights":          {t: round(w * 100, 4) for t, w in top5},
            "all_weights":           {t: round(w, 6) for t, w in list(weights.items())},
        }

    elif name == "get_regime_summary":
        csv = OUTPUT_DIR / "regime_predictions.csv"
        if not csv.exists():
            return {"status": "error", "message": "ยังไม่มีข้อมูล กรุณารัน classify_market_regime ก่อน"}
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        latest = df.iloc[-1]
        return {
            "date":         str(df.index[-1].date()),
            "regime":       latest["regime_label"],
            "prob_bull":    round(float(latest["prob_bull"]), 4),
            "prob_neutral": round(float(latest["prob_neutral"]), 4),
            "prob_bear":    round(float(latest["prob_bear"]), 4),
        }

    elif name == "get_latest_prices":
        csv = OUTPUT_DIR / "prices_clean_thb.csv"
        if not csv.exists():
            return {"status": "error", "message": "ยังไม่มีข้อมูล กรุณารัน data pipeline ก่อน"}
        n = int(args.get("n_rows", 5))
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        latest = df.tail(n).round(2)
        return {
            "source": "output/prices_clean_thb.csv",
            "date_range": f"{latest.index[0].date()} → {latest.index[-1].date()}",
            "as_of": str(latest.index[-1].date()),
            "prices":     latest.iloc[-1].to_dict(),
        }

    elif name == "get_backtest_results":
        from backtest import run_backtest
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: run_backtest(period=args.get("period", "all")))
        return {"source": "prices_clean_thb.csv + regime_predictions.csv + optimal_portfolio.csv",
                **result}

    elif name == "get_mpt_analysis":
        csv = OUTPUT_DIR / "optimal_portfolio.csv"
        if not csv.exists():
            return {"status": "error", "message": "ยังไม่มีผล MPT ในระบบ"}
        row = pd.read_csv(csv).iloc[0]
        raw = str(row.get("weights", ""))
        import re
        weights = {t: float(v) for t, v in re.findall(r"'([^']+)':\s*(?:np\.float64\()?(-?[\d.eE+]+)", raw)}
        return {
            "source": "optimal_portfolio.csv",
            "as_of": datetime.fromtimestamp(csv.stat().st_mtime).isoformat(),
            "objective": row.get("objective"), "success": bool(row.get("success", False)),
            "annualized_return_pct": float(row.get("annualized_return", 0)),
            "annualized_volatility_pct": float(row.get("annualized_volatility", 0)),
            "sharpe_ratio": float(row.get("sharpe_ratio", 0)),
            "weights_pct": {t: round(w * 100, 2) for t, w in sorted(weights.items(), key=lambda x: -x[1]) if w > 0.0001},
        }

    elif name == "get_regime_analysis":
        result = await execute_tool("get_regime_summary", {})
        if result.get("status") == "error":
            return result
        return {"source": "regime_predictions.csv", **result}

    elif name == "get_live_market_data":
        from live import get_live_snapshot
        try:
            snapshot = await get_live_snapshot()
        except Exception:
            return {"status": "error", "message": "ดึงราคาสดจาก Yahoo Finance ไม่สำเร็จ"}
        requested = [str(t).strip().upper() for t in args.get("tickers", []) if str(t).strip()]
        available = snapshot.get("prices", {})
        selected = requested or sorted(available)
        quotes = {
            ticker: {
                "price_thb": available[ticker],
                "change_pct": snapshot.get("changes", {}).get(ticker),
                "as_of": snapshot.get("asof", {}).get(ticker),
            }
            for ticker in selected if ticker in available
        }
        return {
            "source": "Yahoo Finance API",
            "status": snapshot.get("status", "unknown"),
            "fetched_at": snapshot.get("fetched_at"),
            "fx_thb_per_usd": snapshot.get("fx_thb_per_usd"),
            "quotes": quotes,
            "not_available": [ticker for ticker in selected if ticker not in available],
        }

    elif name == "get_market_news":
        from news import get_news_snapshot
        snapshot = await get_news_snapshot()
        ticker = str(args.get("ticker", "")).strip().upper()
        items = snapshot["items"]
        if ticker:
            items = [item for item in items if item.get("ticker", "").upper() == ticker]
        return {
            "source": "Yahoo Finance News API",
            "status": snapshot["status"],
            "fetched_at": snapshot["fetched_at"],
            "ticker_filter": ticker or None,
            "items": items[:8],
        }

    else:
        return {"status": "error", "message": f"ไม่รู้จัก tool: {name}"}


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "openai_sdk": OPENAI_AVAILABLE,
        "output_files": [f.name for f in OUTPUT_DIR.glob("*.csv")] if OUTPUT_DIR.exists() else [],
    }


@app.post("/api/pipeline")
async def api_pipeline(req: PipelineRequest, bg: BackgroundTasks):
    """รัน data pipeline — ดึงราคาและ clean data"""
    try:
        result = await execute_tool("run_data_pipeline", req.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mpt")
async def api_mpt(req: MPTRequest):
    """รัน MPT optimizer"""
    try:
        result = await execute_tool("optimize_portfolio", req.model_dump())
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="ไม่พบไฟล์ daily_returns.csv — กรุณารัน /api/pipeline ก่อน"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/regime")
async def api_regime(req: RegimeRequest):
    """รัน Market Regime classifier"""
    try:
        result = await execute_tool("classify_market_regime", req.model_dump())
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="ไม่พบไฟล์ prices_clean_thb.csv — กรุณารัน /api/pipeline ก่อน"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/latest")
async def api_portfolio_latest():
    return await execute_tool("get_portfolio_summary", {})


@app.get("/api/regime/latest")
async def api_regime_latest():
    return await execute_tool("get_regime_summary", {})


@app.get("/api/prices")
async def api_prices(n: int = 5):
    return await execute_tool("get_latest_prices", {"n_rows": n})


@app.get("/api/returns-history")
async def api_returns_history(period: str = "1y"):
    """
    คืน cumulative return (%) จริง (rebased เป็น 0% ณ ต้นช่วง) สำหรับ line chart
    period: '1d' | '1w' | '1m' | '3m' | '6m' | '1y' | '3y' | 'all'
    ข้อมูลเป็น end-of-day ต่อวัน — คำนวณด้วย compounding จาก daily returns จริง
    """
    csv = OUTPUT_DIR / "daily_returns.csv"
    if not csv.exists():
        raise HTTPException(
            status_code=404,
            detail="ไม่พบ daily_returns.csv — กรุณารัน /api/pipeline ก่อน"
        )

    df = pd.read_csv(csv, index_col=0, parse_dates=True).sort_index()

    # ใช้ได้ทุกตัว (เลือกแสดงฝั่ง frontend) — เรียงชื่อให้คงที่
    df = df.reindex(sorted(df.columns), axis=1)
    all_tickers = list(df.columns)

    # daily_returns.csv เก็บเป็น decimal (0.01 = 1%) — guard เผื่อเก็บเป็น %
    if not df.empty and df.abs().mean().mean() > 1:
        df = df / 100

    if df.empty:
        return {"period": period, "tickers": [], "data": []}

    # กรองช่วงเวลา (อ้างอิงจากวันข้อมูลล่าสุด ไม่ใช่ today เพื่อไม่ให้ช่วงว่าง)
    now = df.index.max()
    cutoff_map = {
        "1d":  now - pd.Timedelta(days=1),
        "1w":  now - pd.Timedelta(weeks=1),
        "1m":  now - pd.DateOffset(months=1),
        "3m":  now - pd.DateOffset(months=3),
        "6m":  now - pd.DateOffset(months=6),
        "1y":  now - pd.DateOffset(years=1),
        "3y":  now - pd.DateOffset(years=3),
        "all": df.index.min(),
    }
    cutoff = cutoff_map.get(period, cutoff_map["1y"])
    sliced = df[df.index >= cutoff]
    if len(sliced) < 2:
        sliced = df.tail(2)

    # cumulative return จริงด้วย compounding แล้ว rebase ให้เริ่มที่ 0% ณ ต้นช่วง
    # (rebase แต่ละคอลัมน์จากค่าแรกที่ valid — รองรับหุ้นที่เริ่มมีข้อมูลทีหลัง)
    growth = (1 + sliced).cumprod()
    base = {c: growth[c].loc[growth[c].first_valid_index()]
            for c in growth.columns if growth[c].first_valid_index() is not None}
    cum = (growth.div(pd.Series(base), axis=1) - 1) * 100

    # ลดจำนวนจุดให้กราฟอ่านง่าย ตามความยาวของช่วง
    gran = {
        "1d": "D", "1w": "D", "1m": "D", "3m": "D", "6m": "D",
        "1y": "W", "3y": "ME", "all": "ME",
    }.get(period, "W")
    if gran != "D":
        cum = cum.resample(gran).last().dropna(how="all")

    fmt = "%b %y" if gran == "ME" else "%d %b"
    labels = cum.index.strftime(fmt)

    records = []
    for lbl, (_, row) in zip(labels, cum.iterrows()):
        entry = {"date": lbl}
        entry.update({k: round(float(v), 2) for k, v in row.items() if pd.notna(v)})
        records.append(entry)

    return {"period": period, "tickers": all_tickers, "data": records}


def _explicit_portfolio_objective(message: str) -> str | None:
    """Recognize unambiguous optimization requests so they always use MPT."""
    text = message.lower()
    min_vol = bool(re.search(
        r"min[\s_-]*vol|lowest volatility|minimum volatility|volatility.{0,20}(ต่ำ|น้อย)|"
        r"(ต่ำ|น้อย).{0,20}(volatility|ความผันผวน)|ความผันผวน.{0,20}(ต่ำ|น้อย)|เสี่ยงต่ำสุด",
        text,
    ))
    max_return = bool(re.search(
        r"max[\s_-]*return|highest return|ผลตอบแทน.{0,20}(สูงสุด|มากที่สุด)|"
        r"return.{0,20}(สูงสุด|มากที่สุด)", text,
    ))
    max_sharpe = bool(re.search(
        r"max[\s_-]*sharpe|sharpe.{0,20}(สูงสุด|ดีที่สุด)|จัดพอร์ต.{0,30}sharpe",
        text,
    ))
    requested = [name for flag, name in (
        (min_vol, "min_vol"), (max_return, "max_return"), (max_sharpe, "max_sharpe")
    ) if flag]
    return requested[0] if len(requested) == 1 else None


def _is_all_tickers_followup(message: str) -> bool:
    return bool(re.search(r"เอา.{0,8}(ทุกตัว|ทั้งหมด)|ทุกตัว|ทุกหุ้น|หุ้นทั้งหมด|ทั้งหมดเลย", message))


def _has_recent_price_question(history: list[dict]) -> bool:
    for item in history[-8:]:
        if item.get("role") == "user" and re.search(
            r"ราคา|ราคาหุ้น|วิเคราะห์.{0,12}หุ้น|หุ้น.{0,12}ราคา|หุ้น.{0,12}วิเคราะห์",
            str(item.get("content", "")),
        ):
            return True
    return False


def _format_live_market_reply(result: dict) -> str:
    quotes = result.get("quotes") or {}
    if not quotes:
        return "ยังไม่พบราคาหุ้นที่ระบบติดตามในผลจาก Yahoo Finance จึงยังสรุปราคาให้ไม่ได้ครับ"

    lines = []
    for ticker, quote in quotes.items():
        price = quote.get("price_thb")
        if price is None:
            continue
        line = f"{ticker}: ฿{float(price):,.2f}"
        change = quote.get("change_pct")
        if change is not None:
            line += f" ({float(change):+.2f}%)"
        if quote.get("as_of"):
            line += f" ณ {quote['as_of']}"
        lines.append(line)

    if not lines:
        return "ผลจาก Yahoo Finance ไม่มีตัวเลขราคาที่ใช้สรุปได้ครับ"
    stale_note = " สถานะข้อมูลระบุว่าเก่า โปรดดูวันกำกับแต่ละรายการ" if result.get("status") == "stale" else ""
    return (
        "ราคาล่าสุดที่ดึงได้จาก Yahoo Finance (แสดงเป็น THB และเปอร์เซ็นต์เปลี่ยนแปลงตามผล API):\n"
        + "\n".join(lines)
        + stale_note
    )


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    return await _api_chat(req, None)


async def chat_with_portfolio(req: ChatRequest, portfolio_context: dict | None):
    """Chat entry used by authenticated conversation routes after ownership checks."""
    return await _api_chat(req, portfolio_context)


async def _api_chat(req: ChatRequest, portfolio_context: dict | None):
    """
    LLM chat endpoint พร้อม Function Calling
    - คำตอบต้องอ้างอิง tool result เท่านั้น; ถ้าไม่มีหลักฐานให้ปฏิเสธการเดา
    """
    tool_calls_log:   list[dict] = []
    tool_results_log: list[dict] = []
    # Resolve a short follow-up like “เอาทุกตัว” against the prior price question.
    # This avoids the model misrouting it to the data pipeline instead of quotes.
    if _is_all_tickers_followup(req.message) and _has_recent_price_question(req.history):
        args = {}
        result = await execute_tool("get_live_market_data", args)
        tool_calls_log.append({"tool": "get_live_market_data", "args": args})
        tool_results_log.append({"tool": "get_live_market_data", "result": result})
        reply = (
            "ดึงราคาหุ้นทั้งหมดที่ระบบติดตามไม่สำเร็จ จึงยังยืนยันราคาไม่ได้ครับ"
            if result.get("status") == "error"
            else _format_live_market_reply(result)
        )
        return ChatResponse(reply=reply, tool_calls=tool_calls_log, tool_results=tool_results_log)

    objective = _explicit_portfolio_objective(req.message)
    if objective:
        args = {"objective": objective}
        result = await execute_tool("optimize_portfolio", args)
        tool_calls_log.append({"tool": "optimize_portfolio", "args": args})
        tool_results_log.append({"tool": "optimize_portfolio", "result": result})
        if result.get("status") == "error":
            reply = result.get("message", "MPT คำนวณไม่สำเร็จ")
        else:
            names = {
                "min_vol": "Min Volatility",
                "max_return": "Max Return",
                "max_sharpe": "Max Sharpe",
            }
            weights = result.get("top5_weights", {})
            allocation = ", ".join(f"{ticker} {weight:.2f}%" for ticker, weight in weights.items())
            reply = (
                f"ผลคำนวณพอร์ต {names[objective]} จากผลตอบแทนย้อนหลังถึง {result.get('as_of', 'ไม่ระบุวันที่')} "
                f"(หน่วยผลตอบแทน/ความผันผวนเป็นรายปี): Return {result['annualized_return']:.2f}%, "
                f"Volatility {result['annualized_volatility']:.2f}%, Sharpe {result['sharpe_ratio']:.4f}. "
                f"น้ำหนักสูงสุด: {allocation}. ผลย้อนหลังไม่รับประกันผลในอนาคต"
            )
        return ChatResponse(reply=reply, tool_calls=tool_calls_log, tool_results=tool_results_log)

    # Let a key in .env select its matching provider unless the user chose one.
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "gemini" if os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY") else "openai"
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        model = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
        provider_name = "Gemini"
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        provider_name = "OpenAI"
    else:
        return ChatResponse(
            reply="ตั้งค่า LLM_PROVIDER ไม่ถูกต้อง ใช้ gemini หรือ openai",
            tool_calls=[], tool_results=[])

    # OpenAI SDK is used as the compatible client for OpenAI and Gemini.
    if OPENAI_AVAILABLE and api_key:
        try:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=165.0)
            system_prompt = (
                "คุณคือผู้ช่วยของ Quantix ตอบภาษาไทยเสมอ. "
                "ตอบคำถามทั่วไปและอธิบายแนวคิดการลงทุนด้วยความรู้ทั่วไปได้ โดยบอกข้อจำกัดเมื่อเหมาะสม. "
                "ข้อมูลเฉพาะของผู้ใช้ พอร์ต ราคา ข่าวล่าสุด และตัวเลข/สถานะของระบบ ตอบได้เฉพาะเมื่อมี tool result รองรับเท่านั้น. "
                "คำถามเกี่ยวกับพอร์ตส่วนตัว หุ้นที่ถือ จำนวนหุ้น วันที่ซื้อ หรือราคาซื้อ ให้เรียก get_my_portfolio เสมอ. "
                "ห้ามเดาตัวเลขหรือสถานะของข้อมูลระบบ. "
                "เมื่อผู้ใช้ขอ Min Volatility ให้เรียก optimize_portfolio ด้วย objective=min_vol; "
                "เมื่อขอผลตอบแทนสูงสุดใช้ max_return; เมื่อขอ Sharpe สูงสุดใช้ max_sharpe. "
                "ห้ามสรุปว่าเป้าหมายเหล่านี้ไม่มีในระบบเพียงเพราะ get_portfolio_summary แสดงพอร์ตหลักเป้าหมายเดียว. "
                "ถ้าถามราคาปัจจุบันหรือการเปลี่ยนแปลง ให้เรียก get_live_market_data; ถ้าถามข่าวล่าสุดให้เรียก get_market_news. "
                "หากข้อความสั้น เช่น 'เอาทุกตัว' ให้อ่านบริบทก่อนหน้าและเรียกเครื่องมือให้ตรงหัวข้อเดิม; ถ้าหมายถึงราคาหุ้นทั้งหมดที่ระบบติดตาม ให้เรียก get_live_market_data โดยไม่ส่ง tickers. "
                "วิเคราะห์ข่าวได้เฉพาะจากหัวข้อ/สรุปที่ tool ส่งมา และบอกให้ชัดว่าราคาเป็นราคาปิดล่าสุด ไม่ใช่ราคา tick real-time. "
                "หากข้อมูลมี status=stale ให้ระบุว่าเป็นข้อมูลเก่า; ห้ามเรียกข้อมูล CSV ว่าราคาสด. "
                "ถ้าไม่มี tool ที่ตรงคำถาม, tool error, หรือข้อมูลไม่ครอบคลุม ให้บอกชัดว่าไม่มีข้อมูลในระบบและไม่สรุปเอง. "
                "ระบุวัน/ช่วงข้อมูลและ source เมื่อมีในผลลัพธ์. แยกคำอธิบายวิธีคำนวณออกจากข้อเท็จจริง. "
                "ผลตอบแทนย้อนหลังหรือผล backtest ไม่ใช่การรับประกันอนาคต."
            )
            messages = [{"role": "system", "content": system_prompt}]
            llm_tools = list(TOOLS)
            if portfolio_context is not None:
                llm_tools.append({
                    "type": "function",
                    "function": {
                        "name": "get_my_portfolio",
                        "description": "อ่านข้อมูลจริงของพอร์ตที่ผู้ใช้เลือกใน My Portfolio รวม ticker, จำนวนหุ้น, วันที่ซื้อ, ราคาซื้อต่อหุ้น และต้นทุนรวม (THB)",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                })
            # Never let client-supplied history add system/developer/tool messages.
            for item in req.history[-20:]:
                role = item.get("role")
                content = item.get("content")
                if role in {"user", "assistant"} and isinstance(content, str):
                    messages.append({"role": role, "content": content[:8000]})
            messages.append({"role": "user", "content": req.message})

            resp = await client.chat.completions.create(
                model=model, messages=messages,
                tools=llm_tools, tool_choice="auto", max_tokens=512,
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments or "{}")
                    tool_calls_log.append({"tool": fn_name, "args": fn_args})
                    if fn_name == "get_my_portfolio":
                        tool_result = {
                            "status": "success",
                            "source": "My Portfolio (บัญชีผู้ใช้ปัจจุบัน)",
                            "as_of": "ข้อมูลที่บันทึกไว้ล่าสุด",
                            **portfolio_context,
                        }
                    else:
                        tool_result = await execute_tool(fn_name, fn_args)
                    tool_results_log.append({"tool": fn_name, "result": tool_result})
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })
                failed_tools = [tr for tr in tool_results_log
                                if tr["result"].get("status") == "error"]
                if failed_tools:
                    names = ", ".join(tr["tool"] for tr in failed_tools)
                    return ChatResponse(
                        reply=f"ดึงข้อมูลจากระบบไม่สำเร็จ ({names}) จึงยังสรุปคำตอบนี้ไม่ได้",
                        tool_calls=tool_calls_log, tool_results=tool_results_log)
                # Tool errors are evidence of missing data, never grounds to improvise.
                resp2 = await client.chat.completions.create(
                    model=model, messages=messages, max_tokens=512,
                )
                final_reply = (resp2.choices[0].message.content or "").strip()
                if not final_reply:
                    live_result = next(
                        (item["result"] for item in tool_results_log
                         if item["tool"] == "get_live_market_data"),
                        None,
                    )
                    final_reply = (
                        _format_live_market_reply(live_result)
                        if live_result is not None
                        else "เครื่องมือดึงข้อมูลสำเร็จ แต่โมเดลไม่ได้ส่งข้อความสรุปกลับมา ผลข้อมูลแสดงในการ์ดด้านล่างครับ"
                    )
            else:
                # For general questions the model can answer without a data tool.
                # Keep personalized/system facts guarded by the tool-only rules above.
                final_reply = (msg.content or "ขออภัยครับ ยังเรียบเรียงคำตอบไม่ได้ ลองถามอีกครั้งได้ไหม")

            return ChatResponse(reply=final_reply,
                                tool_calls=tool_calls_log,
                                tool_results=tool_results_log)

        except Exception as llm_err:
            import logging
            # Provider exceptions may include a masked fragment of the API key.
            logging.warning("LLM request failed (%s)", type(llm_err).__name__)
            if type(llm_err).__name__ == "AuthenticationError":
                return ChatResponse(
                    reply=f"{provider_name} ปฏิเสธ API key (401) กรุณาตรวจว่า key ในไฟล์ .env ถูกต้อง ยังใช้งานได้ และไม่มีข้อความตัวอย่าง/อัญประกาศ จากนั้นสร้าง backend ใหม่",
                    tool_calls=tool_calls_log, tool_results=tool_results_log)
            if type(llm_err).__name__ == "RateLimitError":
                return ChatResponse(
                    reply=f"{provider_name} จำกัดคำขอ (429) อาจถึง rate limit, quota หรือ billing limit กรุณาตรวจสอบ usage และ limits ของ API key นี้",
                    tool_calls=tool_calls_log, tool_results=tool_results_log)
            return ChatResponse(
                reply="ขณะนี้ระบบวิเคราะห์คำถามไม่ได้ จึงยังยืนยันข้อมูลให้ไม่ได้ กรุณาลองใหม่ภายหลัง",
                tool_calls=tool_calls_log, tool_results=tool_results_log)

    # No LLM: fail closed instead of keyword guessing or presenting raw results as an answer.
    return ChatResponse(
        reply=f"ยังตอบคำถามจากข้อมูลระบบไม่ได้ กรุณาตั้งค่า {('GEMINI_API_KEY' if provider == 'gemini' else 'OPENAI_API_KEY')} "
              "และตรวจว่า openai SDK ติดตั้งแล้ว; ระบบจะไม่เดาคำตอบแทนข้อมูล",
        tool_calls=[], tool_results=[])


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
