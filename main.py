import os
import json
import logging
import asyncio
import time
import string
import base64
from typing import List, Dict, Set
from dotenv import load_dotenv
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from openai import AsyncOpenAI
import websockets
from asyncio import Queue
import html

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("main")

# =====================================================
# ENV
# =====================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MEMO_API_KEY = os.getenv("MEMO_API_KEY", "").strip()
NOTION_API_KEY = os.getenv("NOTION_API_KEY", "").strip()
NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID", "").strip()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()  # kept for compatibility, unused now
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"

# Feature flags / tuning via env
USE_SSML = os.getenv("USE_SSML", "1") == "1"
CHUNK_CHAR_THRESHOLD = int(os.getenv("CHUNK_CHAR_THRESHOLD", "20"))  # lower -> start TTS earlier
PUNCTUATE_WITH_LLM = os.getenv("PUNCTUATE_WITH_LLM", "0") == "1"

# =====================================================
# n8n ENDPOINTS
# =====================================================
N8N_CALENDAR_URL = "https://n8n.marshall321.org/webhook/calendar-agent"
N8N_PLATE_URL = "https://n8n.marshall321.org/webhook/agent/plate"

# =====================================================
# MODEL
# =====================================================
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = "gpt-5.1"

# =====================================================
# FASTAPI
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
# MEM0 HELPERS
# =====================================================
async def mem0_search(user_id: str, query: str):
    if FAST_MODE:
        return []
    if not MEMO_API_KEY:
        return []
    headers = {"Authorization": f"Token {MEMO_API_KEY}"}
    payload = {"filters": {"user_id": user_id}, "query": query}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post("https://api.mem0.ai/v2/memories/", headers=headers, json=payload)
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
# NOTION PROMPT
# =====================================================
async def get_notion_prompt():
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
                    parts.append("".join([t.get("plain_text", "") for t in blk["paragraph"]["rich_text"]]))
            return "\n".join(parts).strip() or "You are Solomon Roth’s AI assistant, Silas."
    except Exception as e:
        log.error(f"❌ Notion error: {e}")
        return "You are Solomon Roth’s AI assistant, Silas."

@app.get("/prompt", response_class=PlainTextResponse)
async def get_prompt_text():
    txt = await get_notion_prompt()
    return PlainTextResponse(txt, headers={"Access-Control-Allow-Origin": "*"})

# =====================================================
# NORMALIZATION
# =====================================================
def _normalize(m: str):
    m = m.lower().strip()
    m = "".join(ch for ch in m if ch not in string.punctuation)
    return " ".join(m.split())

def _is_similar(a: str, b: str):
    return bool(a and b and (a == b or a.startswith(b) or b.startswith(a) or a in b or b in a))

# =====================================================
# Utility: prepare TTS input with light SSML
# =====================================================
def escape_for_ssML(s: str) -> str:
    return html.escape(s, quote=False)

def make_ssml_from_text(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    t_esc = escape_for_ssML(t)
    return f'<speak><prosody rate="1.30">{t_esc}</prosody></speak>'

# =====================================================
# WEBSOCKET HANDLER
# =====================================================
@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):
    await ws.accept()

    user_id = "solomon_roth"
    processed_messages = set()

    chat_history: List[Dict] = []
    turn_id = 0
    current_active_turn_id = 0

    calendar_kw = ["calendar", "meeting", "schedule", "appointment"]
    plate_kw = ["plate", "add", "to-do", "task", "notion", "list"]

    prompt = await get_notion_prompt()
    greet = prompt.splitlines()[0] if prompt else "Hello Solomon, I’m Silas."

    # Greeting
    try:
        log.info("👋 Sending greeting TTS")
        greet_input = make_ssml_from_text(greet) if USE_SSML else greet
        tts_greet = await openai_client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=greet_input
        )
        await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": 0}))
        await ws.send_bytes(await tts_greet.aread())
    except Exception as e:
        log.error(f"❌ Greeting TTS error: {e}")

    if not OPENAI_API_KEY:
        log.error("❌ No OPENAI_API_KEY set.")
        return

    rt_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"

    try:
        log.info("🌐 Connecting to OpenAI Realtime...")
        rt_ws = await websockets.connect(
            rt_url,
            additional_headers=[
                ("Authorization", f"Bearer {OPENAI_API_KEY}"),
                ("OpenAI-Beta", "realtime=v1")
            ],
            ping_interval=None,
            max_size=None,
            close_timeout=0
        )
        log.info("✅ Connected to OpenAI Realtime")
        await rt_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "input_audio": {
                    "format": "pcm16",
                    "sample_rate": 48000
                },
                "output_audio": {
                    "format": "wav"
                }
            }
        }))
    except Exception as e:
        log.error(f"❌ Failed to connect to OpenAI Realtime WS: {e}")
        return

    incoming_audio_queue: Queue = Queue()
    tts_tasks_by_turn: Dict[int, Set[asyncio.Task]] = {}
    last_audio_time = time.time()

    # --- Realtime listener with verbose logging ---
    async def realtime_listener_task():
        nonlocal turn_id, current_active_turn_id
        try:
            async for raw in rt_ws:
                try:
                    raw_text = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else raw
                    data = json.loads(raw_text)
                    if not isinstance(data, dict):
                        continue

                    evt_type = data.get("type", "")
                    log.info(f"Realtime event: {evt_type} payload_keys={list(data.keys())} payload={data}")

                    if evt_type == "response.created":
                        turn_id += 1
                        current_active_turn_id = turn_id
                        try:
                            await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_active_turn_id}))
                        except Exception:
                            pass
                        continue

                    if evt_type in ("response.output_audio.delta", "response.audio.delta"):
                        audio_b64 = data.get("delta") or data.get("audio") or ""
                        if audio_b64:
                            try:
                                audio_bytes = base64.b64decode(audio_b64)
                                await ws.send_bytes(audio_bytes)
                            except Exception as e:
                                log.error(f"❌ Error sending audio delta to client: {e}")
                        continue

                    if evt_type and evt_type.startswith("error"):
                        log.error(f"❌ Realtime error event: {data}")
                        continue

                except Exception as e:
                    log.error(f"❌ Realtime parse error: {e}")
        except Exception as e:
            log.error(f"❌ Realtime listener fatal: {e}")

    asyncio.create_task(realtime_listener_task())

    # Reader loop
    async def ws_reader():
        nonlocal last_audio_time, turn_id, current_active_turn_id, audio_accum, last_commit_ts
        try:
            while True:
                msg = await ws.receive()
                if msg is None:
                    break
                mtype = msg.get("type")
                if mtype == "websocket.receive":
                    if "text" in msg and msg["text"] is not None:
                        try:
                            data = json.loads(msg["text"])
                        except Exception:
                            continue
                        typ = data.get("type")
                        if typ == "interrupt":
                            turn_id += 1
                            current_active_turn_id = turn_id
                            audio_accum.clear()
                            log.info(f"⏹️ Received interrupt from client — new active turn {current_active_turn_id}")
                            for t_id, tasks in list(tts_tasks_by_turn.items()):
                                if t_id != current_active_turn_id:
                                    for t in tasks:
                                        try:
                                            t.cancel()
                                        except Exception:
                                            pass
                                    tts_tasks_by_turn.pop(t_id, None)
                        else:
                            log.debug(f"WS text message (ignored): {data}")
                    elif "bytes" in msg and msg["bytes"] is not None:
                        audio_bytes = msg["bytes"]
                        log.info(f"🎤 Received audio frame from client, len={len(audio_bytes)} bytes")
                        await incoming_audio_queue.put(audio_bytes)
                        last_audio_time = time.time()
                elif mtype == "websocket.disconnect":
                    log.info("WS reader noticed disconnect")
                    break
        except WebSocketDisconnect:
            log.info("WS reader disconnected")
        except Exception as e:
            log.error(f"ws_reader fatal: {e}")

    reader_task = asyncio.create_task(ws_reader())

    # Audio sender with min-bytes AND max-wait guard
    min_bytes_per_commit = int(48000 * 2 * 0.10)  # 100ms of 48k PCM16 mono = 9600 bytes
    max_wait_seconds = 0.25  # flush whatever we have every 250ms if non-empty
    audio_accum = bytearray()
    last_commit_ts = time.time()

    async def realtime_audio_sender():
        nonlocal last_commit_ts
        try:
            while True:
                data = await incoming_audio_queue.get()
                if data is None:
                    break
                audio_accum.extend(data)

                now = time.time()
                should_flush = len(audio_accum) >= min_bytes_per_commit or (
                    len(audio_accum) > 0 and (now - last_commit_ts) >= max_wait_seconds
                )
                if not should_flush:
                    continue

                try:
                    if len(audio_accum) == 0:
                        continue
                    chunk = bytes(audio_accum)
                    audio_accum.clear()
                    audio_b64 = base64.b64encode(chunk).decode("utf-8")
                    log.info(f"↗️ Sending audio to Realtime: len={len(chunk)} bytes")
                    await rt_ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
                    await rt_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    await rt_ws.send(json.dumps({"type": "response.create", "response": {}}))
                    last_commit_ts = now
                except Exception as e:
                    log.error(f"❌ Error sending audio to Realtime WS: {e}")
        except asyncio.CancelledError:
            return

    rt_sender_task = asyncio.create_task(realtime_audio_sender())

    # Cleanup
    try:
        await reader_task
    except Exception:
        pass
    finally:
        try:
            reader_task.cancel()
        except Exception:
            pass
        try:
            rt_sender_task.cancel()
        except Exception:
            pass
        for tasks in tts_tasks_by_turn.values():
            for t in tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
        try:
            await rt_ws.close()
        except Exception:
            pass

# =====================================================
# helper for n8n calls (same as before)
# =====================================================
async def send_to_n8n(url: str, text: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(url, json={"text": text})
            if r.status_code == 200:
                return r.text
    except Exception as e:
        log.error(f"n8n error: {e}")
    return "Okay."

# =====================================================
# SERVER START
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
