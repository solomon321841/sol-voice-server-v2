import os
import json
import logging
import asyncio
import time
import random
import string
from typing import List, Dict
from dotenv import load_dotenv
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from openai import AsyncOpenAI

# =====================================================
# 🔧 LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# =====================================================
# 🔑 ENV
# =====================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MEMO_API_KEY = os.getenv("MEMO_API_KEY", "").strip()
NOTION_API_KEY = os.getenv("NOTION_API_KEY", "").strip()
NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID", "").strip()

# =====================================================
# 🌐 n8n ENDPOINTS (Notion & Mem0 calls included)
# =====================================================
N8N_CALENDAR_URL = "https://n8n.marshall321.org/webhook/calendar-agent"
N8N_PLATE_URL = "https://n8n.marshall321.org/webhook/agent/plate"

# =====================================================
# 🤖 MODEL
# =====================================================
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = "gpt-4o-mini"

# =====================================================
# ⚙️ FASTAPI APP
# =====================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def home():
    return {"status": "running", "message": "Silas backend is online."}

@app.get("/health")
async def health():
    return {"ok": True}

# =====================================================
# 🧠 MEM0 MEMORY (merged from prior version)
# =====================================================
async def mem0_search(user_id: str, query: str):
    if not MEMO_API_KEY:
        return []
    headers = {"Authorization": f"Token {MEMO_API_KEY}"}
    payload = {"filters": {"user_id": user_id}, "query": query}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post("https://api.mem0.ai/v2/memories/", headers=headers, json=payload)
            if r.status_code == 200:
                return r.json() if isinstance(r.json(), list) else []
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
            await c.post("https://api.mem0.ai/v1/memories/", headers=headers, json=payload)
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
# 🧩 NOTION PROMPT (merged from prior version)
# =====================================================
async def get_notion_prompt():
    if not NOTION_PAGE_ID or not NOTION_API_KEY:
        return "You are Solomon Roth’s personal AI assistant, Silas."
    url = f"https://api.notion.com/v1/blocks/{NOTION_PAGE_ID}/children"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
            parts = []
            for blk in data.get("results", []):
                if blk.get("type") == "paragraph":
                    txt = "".join([r.get("plain_text", "") for r in blk["paragraph"]["rich_text"]])
                    parts.append(txt)
            return "\n".join(parts).strip() or "You are Solomon Roth’s AI assistant, Silas."
    except Exception as e:
        log.error(f"❌ Notion error: {e}")
        return "You are Solomon Roth’s AI assistant, Silas."

# =====================================================
# 🔹 /prompt ENDPOINT
# =====================================================
@app.get("/prompt", response_class=PlainTextResponse)
async def get_prompt_text():
    text = await get_notion_prompt()
    headers = {"Access-Control-Allow-Origin": "*"}
    return PlainTextResponse(text, headers=headers)

# =====================================================
# 🧩 n8n HELPERS
# =====================================================
async def send_to_n8n(url: str, message: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            payload = {"message": message}
            r = await c.post(url, json=payload)
            log.info(f"📩 n8n raw response: {r.text}")

            if r.status_code == 200:
                try:
                    data = r.json()
                    if isinstance(data, dict):
                        return (
                            data.get("reply")
                            or data.get("message")
                            or data.get("text")
                            or data.get("output")
                            or json.dumps(data, indent=2)
                        ).strip()
                    if isinstance(data, list):
                        return " ".join(str(x) for x in data)
                    return str(data)
                except:
                    return r.text.strip()
            return "Sorry, the automation returned an unexpected response."

    except Exception as e:
        log.error(f"n8n error: {e}")
        return "Sorry, couldn't reach automation."

# =====================================================
# 🔌 RETELL WS — Connection + Debounce Fix
# =====================================================
connections = {}

def _normalize(msg: str):
    msg = msg.lower().strip()
    msg = "".join(ch for ch in msg if ch not in string.punctuation)
    msg = " ".join(msg.split())
    return msg

def _is_similar(a: str, b: str):
    if not a or not b:
        return False
    if a == b:
        return True
    if a.startswith(b) or b.startswith(a):
        return True
    if a in b or b in a:
        return True
    return False

@app.websocket("/ws/{call_id}")
async def ws_handler(ws: WebSocket, call_id: str):

    # CLEAN ALL OTHER CONNECTIONS
    for cid, conn in list(connections.items()):
        try:
            conn["active"] = False
            await conn["ws"].close()
        except:
            pass
        connections.pop(cid, None)

    await ws.accept()
    connections[call_id] = {"ws": ws, "active": True}
    user_id = "solomon_roth"

    async def speak(resp_id, text, end=True):
        if not connections.get(call_id, {}).get("active"):
            return
        payload = {
            "type": "response_message",
            "response_id": resp_id,
            "content": text,
            "content_complete": end,
            "end_turn": end,
        }
        try:
            await ws.send_text(json.dumps(payload))
        except:
            pass

    # GREETING (uses merged Notion prompt)
    prompt = await get_notion_prompt()
    greet = prompt.splitlines()[0] if prompt else "Hello Solomon, I’m Silas."
    await speak(0, greet)

    # === DEBOUNCE FIX ===
    recent_msgs = []
    processed_messages = set()     # <— duplicate blocker

    # Keywords & phrases
    calendar_kw = ["calendar", "meeting", "schedule", "appointment"]
    plate_kw = ["plate", "add", "to-do", "task", "notion", "list"]
    plate_add_kw = ["add", "put", "create", "new", "include"]
    plate_check_kw = ["what", "show", "see", "check", "read"]

    add_phrases = [
        "Of course boss. Doing that now.",
        "Gotcha. Give me one sec.",
        "Of course. Adding that now.",
        "Okay. Putting that on your plate.",
        "Not a problem. I’ll be right back.",
    ]
    check_phrases = [
        "Let’s see what’s on your plate...",
        "One moment, checking that for you...",
        "Alright, here’s what you’ve got...",
        "Give me a sec, pulling that up...",
    ]
    calendar_phrases = [
        "Let me check your schedule real quick...",
        "Just a second while I pull that up...",
        "Alright, let’s take a look at your calendar...",
        "Okay, seeing what’s on your agenda...",
    ]

    try:
        while True:
            raw = await ws.receive_text()

            if not connections.get(call_id, {}).get("active"):
                break

            data = json.loads(raw)
            trans = data.get("transcript", [])
            inter = data.get("interaction_type")
            rid = data.get("response_id", 1)

            msg = ""
            for t in reversed(trans or []):
                if t.get("role") == "user":
                    msg = t.get("content", "")
                    break

            if not msg or inter != "response_required":
                continue

            # === NORMAL DEBOUNCE ===
            norm = _normalize(msg)
            now = time.time()
            recent_msgs = [(m, ts) for (m, ts) in recent_msgs if now - ts < 2]
            if any(_is_similar(m, norm) for (m, ts) in recent_msgs):
                log.info(f"🛑 Skipping duplicate / partial-like message: {msg}")
                continue
            recent_msgs.append((norm, now))

            # MEMORY (uses merged Mem0 functions)
            mems = await mem0_search(user_id, msg)
            ctx = memory_context(mems)
            sys_prompt = f"{prompt}\n\nFacts:\n{ctx}"
            lower_msg = msg.lower()

            # =========================================================
            # PLATE — WITH STRONG DUPLICATE BLOCKER
            # =========================================================
            if any(k in lower_msg for k in plate_kw):

                if msg in processed_messages:
                    log.info(f"🛑 HARD BLOCK duplicate send_to_n8n: {msg}")
                    continue
                processed_messages.add(msg)

                if any(k in lower_msg for k in plate_add_kw):
                    phrase = random.choice(add_phrases)
                elif any(k in lower_msg for k in plate_check_kw):
                    phrase = random.choice(check_phrases)
                else:
                    phrase = "Let me handle that..."

                await speak(rid, phrase, end=False)
                reply = await send_to_n8n(N8N_PLATE_URL, msg)
                await speak(rid, reply)
                continue

            # CALENDAR
            if any(k in lower_msg for k in calendar_kw):
                await speak(rid, random.choice(calendar_phrases), end=False)
                reply = await send_to_n8n(N8N_CALENDAR_URL, msg)
                await speak(rid, reply)
                continue

            # DEFAULT CHAT
            try:
                stream = await openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": msg},
                    ],
                    stream=True,
                )
                async for chunk in stream:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        await speak(rid, delta, end=False)
                await speak(rid, "", end=True)
                asyncio.create_task(mem0_add(user_id, msg))

            except Exception as e:
                log.error(f"LLM error: {e}")
                await speak(rid, "Sorry, I hit a small issue.")

    except WebSocketDisconnect:
        log.info(f"❌ Retell disconnected {call_id}")

    finally:
        if call_id in connections:
            connections[call_id]["active"] = False
            try:
                await connections[call_id]["ws"].close()
            except:
                pass
            connections.pop(call_id, None)

        log.info(f"🧹 Connection {call_id} fully terminated.")

# =====================================================
# 🚀 RUN
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
