import os
import json
import logging
import asyncio
import time
from typing import Dict, Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from openai import AsyncOpenAI

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("main")

# =====================================================
# ENV
# =====================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MEMO_API_KEY = os.getenv("MEMO_API_KEY", "").strip()
NOTION_API_KEY = os.getenv("NOTION_API_KEY", "").strip()
NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID", "").strip()

N8N_CALENDAR_URL = "https://n8n.marshall321.org/webhook/calendar-agent"
N8N_PLATE_URL = "https://n8n.marshall321.org/webhook/agent/plate"

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

GPT_MODEL = "gpt-5.1"  # for any text tools if needed

# =====================================================
# FASTAPI
# =====================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Netlify origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():
    return {"status": "running", "message": "Silas backend (Realtime tools) is online."}


@app.get("/health")
async def health():
    return {"ok": True}


# =====================================================
# REALTIME TOKEN ENDPOINT
# =====================================================
@app.get("/realtime-token")
async def realtime_token():
    """
    Returns a token the frontend can use to open the OpenAI Realtime WS.
    For now, this is just OPENAI_API_KEY from the environment.
    In production you should ideally mint a short‑lived token instead.
    """
    if not OPENAI_API_KEY:
        return JSONResponse({"error": "OPENAI_API_KEY not configured"}, status_code=500)
    return JSONResponse({"token": OPENAI_API_KEY})


# =====================================================
# MEM0 HELPERS
# =====================================================
async def mem0_search(user_id: str, query: str):
    if not MEMO_API_KEY:
        return []
    headers = {"Authorization": f"Token {MEMO_API_KEY}"}
    payload = {"filters": {"user_id": user_id}, "query": query}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                "https://api.mem0.ai/v2/memories/",
                headers=headers,
                json=payload,
            )
            if r.status_code == 200:
                out = r.json()
                return out if isinstance(out, list) else []
    except Exception as e:
        log.error(f"MEM0 search error: {e}")
    return []


async def mem0_add(user_id: str, text: str):
    if not MEMO_API_KEY or not text:
        return
    headers = {"Authorization": f"Token {MEMO_API_KEY}"}
    payload = {"user_id": user_id, "messages": [{"role": "user", "content": text}]}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(
                "https://api.mem0.ai/v1/memories/",
                headers=headers,
                json=payload,
            )
    except Exception as e:
        log.error(f"MEM0 add error: {e}")


def memory_context(memories: list) -> str:
    if not memories:
        return ""
    lines = []
    for m in memories:
        txt = m.get("memory") or m.get("content") or m.get("text")
        if txt:
            lines.append(f"- {txt}")
    return "Relevant memories:\n" + "\n".join(lines)


# =====================================================
# NOTION PROMPT
# =====================================================
async def get_notion_prompt():
    """
    Fetches your Silas system prompt from a Notion page.
    """
    if not NOTION_PAGE_ID or not NOTION_API_KEY:
        return "You are Solomon Roth’s personal AI assistant, Silas."

    url = f"https://api.notion.com/v1/blocks/{NOTION_PAGE_ID}/children"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
            parts = []
            for blk in data.get("results", []):
                if blk.get("type") == "paragraph":
                    parts.append(
                        "".join(
                            [t.get("plain_text", "") for t in blk["paragraph"]["rich_text"]]
                        )
                    )
            return (
                "\n".join(parts).strip()
                or "You are Solomon Roth’s AI assistant, Silas."
            )
    except Exception as e:
        log.error(f"❌ Notion error: {e}")
        return "You are Solomon Roth’s AI assistant, Silas."


@app.get("/prompt", response_class=PlainTextResponse)
async def get_prompt_text():
    """
    Expose the system prompt as plain text, if you want to use it
    in the Realtime session config or for debugging.
    """
    txt = await get_notion_prompt()
    return PlainTextResponse(txt, headers={"Access-Control-Allow-Origin": "*"})


# =====================================================
# n8n helper (calendar / plate)
# =====================================================
async def send_to_n8n(url: str, msg: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(url, json={"message": msg})
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return data.get("reply") or data.get("message") or json.dumps(data)
            return str(data)
        return f"Error contacting service (status {r.status_code})."
    except Exception as e:
        log.error(f"❌ n8n error calling {url}: {e}")
        return "Sorry, I had an error contacting that service."


# =====================================================
# TOOL ENDPOINTS FOR REALTIME
# =====================================================
@app.post("/tool/plate")
async def tool_plate(body: Dict[str, Any]):
    """
    Tool endpoint for 'plate' tasks, to be called from Realtime (via your frontend).
    Expected JSON:
      { "query": "text the user said" }
    """
    query = (body or {}).get("query") or ""
    log.info(f"🧰 /tool/plate called with query={query!r}")
    reply = await send_to_n8n(N8N_PLATE_URL, query)
    return JSONResponse({"result": reply})


@app.post("/tool/calendar")
async def tool_calendar(body: Dict[str, Any]):
    """
    Tool endpoint for calendar scheduling.
    Expected JSON:
      { "query": "text the user said" }
    """
    query = (body or {}).get("query") or ""
    log.info(f"🧰 /tool/calendar called with query={query!r}")
    reply = await send_to_n8n(N8N_CALENDAR_URL, query)
    return JSONResponse({"result": reply})


@app.post("/tool/memories/search")
async def tool_memories_search(body: Dict[str, Any]):
    """
    Search memories via Mem0.
    Expected JSON:
      { "user_id": "solomon_roth", "query": "text" }
    """
    user_id = (body or {}).get("user_id") or "solomon_roth"
    query = (body or {}).get("query") or ""
    log.info(f"🧰 /tool/memories/search user={user_id!r}, query={query!r}")
    mems = await mem0_search(user_id, query)
    return JSONResponse({"results": mems})


@app.post("/tool/memories/add")
async def tool_memories_add(body: Dict[str, Any]):
    """
    Add a memory via Mem0.
    Expected JSON:
      { "user_id": "solomon_roth", "text": "fact to remember" }
    """
    user_id = (body or {}).get("user_id") or "solomon_roth"
    text = (body or {}).get("text") or ""
    log.info(f"🧰 /tool/memories/add user={user_id!r}, text={text!r}")
    asyncio.create_task(mem0_add(user_id, text))
    return JSONResponse({"status": "queued"})


# =====================================================
# SERVER START
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
