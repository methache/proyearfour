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
import logging
from pathlib import Path
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# โหลด .env ถ้ามี (ไม่ต้องติดตั้ง python-dotenv — อ่านไฟล์เอง)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ─── optional LLM client (OpenAI SDK) ───
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

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
    model:       str  = Field("gpt-4o-mini", description="ชื่อ LLM model")
    api_key:     str | None = Field(None, description="OpenAI API key (ถ้าไม่ใส่ ใช้ ENV)")
    base_url:    str | None = Field(None, description="base_url สำหรับ Ollama หรือ Azure")
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
                "คำนวณพอร์ตการลงทุนที่เหมาะสมที่สุดด้วย Modern Portfolio Theory (MPT) "
                "หา Sharpe Ratio สูงสุด หรือ Volatility ต่ำสุด หรือ Return สูงสุด "
                "พร้อม Efficient Frontier และ Monte Carlo Simulation"
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
            "description": "ดึงผลการคำนวณพอร์ตล่าสุด (น้ำหนักสินทรัพย์, Sharpe, Return, Volatility)",
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
        from mpt import run_mpt
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_mpt(
                objective=args.get("objective", "max_sharpe"),
                top_n=args.get("top_n"),
            )
        )
        opt = result["optimal_portfolio"]
        top5 = sorted(opt["weights"].items(), key=lambda x: -x[1])[:5]
        return {
            "status":                "success",
            "objective":             opt["objective"],
            "sharpe_ratio":          round(float(opt["sharpe_ratio"]), 4),
            "annualized_return":     round(float(opt["annualized_return"]), 4),
            "annualized_volatility": round(float(opt["annualized_volatility"]), 4),
            "top5_weights":          {t: round(w * 100, 4) for t, w in top5},
            "all_weights":           {t: round(w, 6) for t, w in opt["weights"].items()},
            "monte_carlo_best_sharpe": result["monte_carlo_summary"]["best_sharpe_mc"],
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
        if df.empty:
            return {"status": "error", "message": "ไฟล์พอร์ตมีอยู่แต่ไม่มีข้อมูล"}
        row = df.iloc[0].to_dict()
        # parse weights — CSV stored dict with numpy wrappers like {'AAPL': np.float64(0.1)}
        import re
        raw = row.get("weights", "") or ""
        weights = {t: float(v) for t, v in re.findall(r"'([^']+)':\s*(?:np\.float64\()?(-?[\d.]+)", raw)}
        top5 = sorted(weights.items(), key=lambda x: -x[1])[:5]
        return {
            "status":                "success",
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
        if df.empty:
            return {"status": "error", "message": "ไฟล์สภาวะตลาดมีอยู่แต่ไม่มีข้อมูล"}
        latest = df.iloc[-1]
        return {
            "status":       "success",
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
        try:
            n = max(1, min(int(args.get("n_rows", 5)), 365))
        except (TypeError, ValueError):
            n = 5
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        if df.empty or df.shape[1] == 0:
            return {"status": "error", "message": "ไฟล์ราคามีอยู่แต่ไม่มีข้อมูลให้อ่าน"}
        latest = df.tail(n).round(2)
        if latest.empty:
            return {"status": "error", "message": "ไม่มีข้อมูลราคาในช่วงที่ร้องขอ"}
        return {
            "status": "success",
            "date_range": f"{latest.index[0].date()} → {latest.index[-1].date()}",
            "prices": latest.iloc[-1].to_dict(),
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


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    """
    LLM chat endpoint พร้อม Function Calling
    - ถ้ามี API key และ openai SDK → ใช้ LLM จริง
    - ถ้าไม่มี หรือ LLM error → fallback rule-based keyword matching
    """
    tool_calls_log:   list[dict] = []
    tool_results_log: list[dict] = []

    # ── Path A: OpenAI Function Calling ──────────────────────────────
    if OPENAI_AVAILABLE and (req.api_key or os.getenv("OPENAI_API_KEY")):
        try:
            client = AsyncOpenAI(
                api_key=req.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=req.base_url or os.getenv("OPENAI_BASE_URL"),
            )
            system_prompt = (
                "คุณคือ Quant AI Assistant ผู้ช่วยวิเคราะห์ข้อมูลการลงทุนของผู้ใช้ "
                "ตอบเป็นภาษาไทยเสมอ\\n"
                "กฎความถูกต้อง: ใช้เฉพาะข้อมูลจากผู้ใช้หรือผลลัพธ์ Tools ที่ได้รับจริงเท่านั้น "
                "เมื่อถามเกี่ยวกับพอร์ต ราคาสินทรัพย์ ผลตอบแทน หรือสภาวะตลาด ให้เรียก Tool ที่เกี่ยวข้องก่อนตอบ\\n"
                "ห้ามเดาตัวเลข ห้ามสร้างราคา/ผลวิเคราะห์ขึ้นเอง และห้ามอ้างว่าเป็นข้อมูลล่าสุด "
                "หาก Tool ไม่ได้ส่งข้อมูลนั้นกลับมา\\n"
                "หาก Tool ส่ง status='error' หรือข้อความ error ให้แจ้งตรง ๆ ว่าทำงานไม่สำเร็จ "
                "พร้อมอธิบายเหตุผลจากผลลัพธ์ ห้ามแทนข้อมูลที่หายไปด้วยตัวเลขสมมติ "
                "หากไม่ทราบสาเหตุให้บอกว่าไม่ทราบสาเหตุ\\n"
                "ถ้าข้อมูลไม่เพียงพอ ให้บอกว่าข้อมูลไม่เพียงพอและระบุสิ่งที่ขาด "
                "อย่าสรุปเกินกว่าหลักฐานที่ได้รับ"
            )
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(req.history)
            messages.append({"role": "user", "content": req.message})

            resp = await client.chat.completions.create(
                model=req.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), messages=messages,
                tools=TOOLS, tool_choice="auto", max_tokens=1024,
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments or "{}")
                        if not isinstance(fn_args, dict):
                            raise ValueError("Tool arguments ต้องเป็น JSON object")
                    except (json.JSONDecodeError, ValueError) as arg_err:
                        fn_args = {}
                        tool_result = {
                            "status": "error",
                            "message": f"อ่าน arguments ของ Tool ไม่สำเร็จ: {arg_err}",
                        }
                    else:
                        try:
                            tool_result = await execute_tool(fn_name, fn_args)
                            if not isinstance(tool_result, dict):
                                tool_result = {
                                    "status": "error",
                                    "message": "Tool คืนผลลัพธ์ในรูปแบบที่ไม่ถูกต้อง",
                                }
                        except Exception as tool_err:
                            logging.exception("Tool execution failed: %s", fn_name)
                            tool_result = {
                                "status": "error",
                                "message": f"Tool {fn_name} ทำงานไม่สำเร็จ: {type(tool_err).__name__}: {tool_err}",
                            }

                    tool_calls_log.append({"tool": fn_name, "args": fn_args})
                    tool_results_log.append({"tool": fn_name, "result": tool_result})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                    })

                resp2 = await client.chat.completions.create(
                    model=req.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=messages,
                    max_tokens=1024,
                )
                final_reply = resp2.choices[0].message.content or ""
                if not final_reply.strip():
                    final_reply = "ได้รับผลจาก Tool แล้ว แต่ AI ไม่ได้ส่งข้อความตอบกลับ กรุณาลองอีกครั้ง"
            else:
                final_reply = msg.content or "ยังไม่มีคำตอบจาก AI"

            return ChatResponse(reply=final_reply,
                                tool_calls=tool_calls_log,
                                tool_results=tool_results_log)

        except Exception as llm_err:
            import logging
            logging.warning("LLM error → rule-based fallback: %s", llm_err)
            # fall through to Path B

    # ── Path B: Rule-based fallback ──────────────────────────────────
    msg_lower = req.message.lower()
    reply_parts: list[str] = []
    keywords_map = {
        ("pipeline", "data", "update", "refresh",
         "\u0e14\u0e36\u0e07\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25",
         "\u0e2d\u0e31\u0e1e\u0e40\u0e14\u0e17"): "run_data_pipeline",
        ("portfolio", "mpt", "sharpe", "weight", "allocat",
         "\u0e1e\u0e2d\u0e23\u0e4c\u0e15",
         "\u0e19\u0e49\u0e33\u0e2b\u0e19\u0e31\u0e01",
         "\u0e08\u0e31\u0e14\u0e1e\u0e2d\u0e23\u0e4c\u0e15",
         "\u0e25\u0e07\u0e17\u0e38\u0e19"): "optimize_portfolio",
        ("regime", "bull", "bear", "neutral", "market",
         "\u0e2a\u0e20\u0e32\u0e27\u0e30",
         "\u0e15\u0e25\u0e32\u0e14"): "classify_market_regime",
        ("price", "latest", "\u0e23\u0e32\u0e04\u0e32",
         "\u0e25\u0e48\u0e32\u0e2a\u0e38\u0e14",
         "\u0e2b\u0e38\u0e49\u0e19"): "get_latest_prices",
    }
    matched_tools: list[str] = []
    for keywords, tool_name in keywords_map.items():
        if any(k in msg_lower for k in keywords):
            matched_tools.append(tool_name)

    if matched_tools:
        for tool_name in matched_tools:
            try:
                result = await execute_tool(tool_name, {})
                if not isinstance(result, dict):
                    result = {
                        "status": "error",
                        "message": "Tool คืนผลลัพธ์ในรูปแบบที่ไม่ถูกต้อง",
                    }
            except Exception as tool_err:
                logging.exception("Fallback tool execution failed: %s", tool_name)
                result = {
                    "status": "error",
                    "message": f"{type(tool_err).__name__}: {tool_err}",
                }

            tool_calls_log.append({"tool": tool_name, "args": {}})
            tool_results_log.append({"tool": tool_name, "result": result})

            if result.get("status") == "error":
                reply_parts.append(
                    f"**{tool_name} ทำงานไม่สำเร็จ**: "
                    f"{result.get('message', 'ไม่ทราบสาเหตุ')}"
                )
            else:
                reply_parts.append(
                    f"**{tool_name}**: {json.dumps(result, ensure_ascii=False, indent=2, default=str)}"
                )
        reply = "ผลการวิเคราะห์จากระบบ:\\n\\n" + "\\n\\n".join(reply_parts)
    else:
        reply = (
            "สวัสดีครับ! ผมเป็น Quant AI Assistant ช่วยได้เรื่อง:\n"
            "- 📊 ดึงและอัพเดทข้อมูลราคาหุ้น\n"
            "- 💼 จัดพอร์ตด้วย MPT / หา Sharpe Ratio สูงสุด\n"
            "- 📈 วิเคราะห์สภาวะตลาด (Bull/Bear/Neutral)\n"
            "- 💰 ดูราคาหุ้นล่าสุด\n\n"
            "ลองถามว่า: 'จัดพอร์ตให้หน่อย' หรือ 'ตลาดตอนนี้เป็นยังไง'"
        )

    return ChatResponse(reply=reply,
                        tool_calls=tool_calls_log,
                        tool_results=tool_results_log)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
