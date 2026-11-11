"""Authenticated conversation history for the AI chat UI."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from auth import _current_user, db
from main import ChatRequest, chat_with_portfolio


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


async def _user_id(request: Request) -> str:
    user = await _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user["user_id"]


async def _owned_conversation(user_id: str, conversation_id: str) -> dict:
    conversation = await db.chat_conversations.find_one(
        {"id": conversation_id, "user_id": user_id}, {"_id": 0}
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="ไม่พบห้องแชท")
    return conversation


def _conversation_view(item: dict) -> dict:
    return {
        "id": item["id"],
        "title": item.get("title", "แชทใหม่"),
        "bookmarked": bool(item.get("bookmarked", False)),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "last_message_preview": item.get("last_message_preview", ""),
        "message_count": int(item.get("message_count", 0)),
    }


class ConversationUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=80)
    bookmarked: bool | None = None


class ChatMessageIn(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    portfolio_id: str | None = Field(None, max_length=80)


def register(app: FastAPI) -> None:
    @app.on_event("startup")
    async def _chat_indexes() -> None:
        await db.chat_conversations.create_index([("user_id", 1), ("updated_at", -1)])
        await db.chat_messages.create_index(
            [("user_id", 1), ("conversation_id", 1), ("created_at", 1)]
        )

    @app.get("/api/chat/conversations")
    async def list_conversations(request: Request):
        user_id = await _user_id(request)
        cursor = db.chat_conversations.find({"user_id": user_id}, {"_id": 0}).sort(
            [("bookmarked", -1), ("updated_at", -1)]
        ).limit(200)
        return {"conversations": [_conversation_view(item) async for item in cursor]}

    @app.post("/api/chat/conversations")
    async def create_conversation(request: Request):
        user_id = await _user_id(request)
        # Reuse the user's pending blank room so repeated "new chat" clicks
        # never create a stack of conversations that have not been used yet.
        pending = await db.chat_conversations.find_one(
            {"user_id": user_id, "message_count": 0},
            {"_id": 0},
            sort=[("created_at", -1)],
        )
        if pending:
            return _conversation_view(pending)

        now = _now()
        item = {
            "id": f"chat_{uuid.uuid4().hex}",
            "user_id": user_id,
            "title": "แชทใหม่",
            "bookmarked": False,
            "created_at": now,
            "updated_at": now,
            "last_message_preview": "",
            "message_count": 0,
        }
        await db.chat_conversations.insert_one(item)
        return _conversation_view(item)

    @app.get("/api/chat/conversations/{conversation_id}")
    async def get_conversation(request: Request, conversation_id: str):
        user_id = await _user_id(request)
        item = await _owned_conversation(user_id, conversation_id)
        cursor = db.chat_messages.find(
            {"user_id": user_id, "conversation_id": conversation_id}, {"_id": 0}
        ).sort("created_at", 1)
        messages = []
        async for message in cursor:
            messages.append({
                "id": message["id"],
                "role": message["role"],
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls", []),
                "tool_results": message.get("tool_results", []),
                "created_at": message.get("created_at"),
            })
        return {"conversation": _conversation_view(item), "messages": messages}

    @app.patch("/api/chat/conversations/{conversation_id}")
    async def update_conversation(
        request: Request, conversation_id: str, update: ConversationUpdate
    ):
        user_id = await _user_id(request)
        await _owned_conversation(user_id, conversation_id)
        changes = {}
        if update.title is not None:
            title = update.title.strip()
            if not title:
                raise HTTPException(status_code=422, detail="ชื่อห้องแชทห้ามเป็นค่าว่าง")
            changes["title"] = title
        if update.bookmarked is not None:
            changes["bookmarked"] = update.bookmarked
        if changes:
            changes["updated_at"] = _now()
            await db.chat_conversations.update_one(
                {"id": conversation_id, "user_id": user_id}, {"$set": changes}
            )
        current = await _owned_conversation(user_id, conversation_id)
        return _conversation_view(current)

    @app.delete("/api/chat/conversations/{conversation_id}")
    async def delete_conversation(request: Request, conversation_id: str):
        user_id = await _user_id(request)
        await _owned_conversation(user_id, conversation_id)
        await db.chat_messages.delete_many(
            {"user_id": user_id, "conversation_id": conversation_id}
        )
        await db.chat_conversations.delete_one(
            {"id": conversation_id, "user_id": user_id}
        )
        return {"ok": True}

    @app.post("/api/chat/conversations/{conversation_id}/messages")
    async def send_message(
        request: Request, conversation_id: str, body: ChatMessageIn
    ):
        user_id = await _user_id(request)
        conversation = await _owned_conversation(user_id, conversation_id)
        # Resolve the requested portfolio against this authenticated user's data.
        # If no selection was sent, use their first (default) portfolio.
        from portfolio_manager import _find_portfolio, _get_or_init_portfolios
        portfolios = await _get_or_init_portfolios(user_id)
        selected = (
            _find_portfolio(portfolios, body.portfolio_id)
            if body.portfolio_id else (portfolios[0] if portfolios else None)
        )
        if body.portfolio_id and not selected:
            raise HTTPException(status_code=404, detail="ไม่พบพอร์ตที่เลือก")
        portfolio_context = None
        if selected:
            portfolio_context = {
                "portfolio_id": selected.get("id"),
                "portfolio_name": selected.get("name", "พอร์ตหลัก"),
                "holdings": [
                    {
                        "ticker": str(item.get("ticker", "")).upper(),
                        "shares": item.get("shares"),
                        "buy_date": item.get("buyDate"),
                        "buy_price_thb_per_share": item.get("buyPrice"),
                        "buy_cost_thb": round(float(item.get("shares") or 0) * float(item.get("buyPrice") or 0), 2),
                    }
                    for item in selected.get("items", [])
                ],
            }
        now = _now()

        prior_cursor = db.chat_messages.find(
            {"user_id": user_id, "conversation_id": conversation_id},
            {"_id": 0, "role": 1, "content": 1},
        ).sort("created_at", -1).limit(20)
        prior = [item async for item in prior_cursor]
        prior.reverse()

        user_message_id = f"msg_{uuid.uuid4().hex}"
        await db.chat_messages.insert_one({
            "id": user_message_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "role": "user",
            "content": body.message,
            "tool_calls": [],
            "tool_results": [],
            "created_at": now,
        })

        try:
            response = await chat_with_portfolio(
                ChatRequest(message=body.message, history=prior), portfolio_context
            )
            reply = response.reply.strip() or "ข้อมูลจากเครื่องมือแสดงด้านล่าง แต่โมเดลไม่ได้ส่งข้อความสรุปกลับมาครับ"
            tool_calls = response.tool_calls
            tool_results = response.tool_results
        except asyncio.CancelledError:
            await db.chat_messages.delete_one(
                {"id": user_message_id, "user_id": user_id, "conversation_id": conversation_id}
            )
            raise
        except Exception:
            reply = "ขณะนี้ระบบตอบข้อความนี้ไม่ได้ กรุณาลองใหม่"
            tool_calls = []
            tool_results = []

        assistant_time = _now()
        await db.chat_messages.insert_one({
            "id": f"msg_{uuid.uuid4().hex}",
            "user_id": user_id,
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": reply,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "created_at": assistant_time,
        })

        title = conversation.get("title", "แชทใหม่")
        if title == "แชทใหม่":
            title = body.message.strip().replace("\n", " ")[:56] or "แชทใหม่"
        await db.chat_conversations.update_one(
            {"id": conversation_id, "user_id": user_id},
            {"$set": {
                "title": title,
                "updated_at": assistant_time,
                "last_message_preview": reply[:180],
            }, "$inc": {"message_count": 2}},
        )
        return {
            "reply": reply,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "conversation": {
                **_conversation_view({**conversation, "title": title,
                                     "updated_at": assistant_time,
                                     "last_message_preview": reply[:180],
                                     "message_count": conversation.get("message_count", 0) + 2}),
            },
        }
