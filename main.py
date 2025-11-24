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
import tempfile
import websockets
from asyncio import Queue   # ✅ ADDED

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
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()

# =====================================================
# 🌐 n8n ENDPOINTS
# =====================================================
N8N_CALENDAR_URL = "https://n8n.marshall321.org/webhook/calendar-agent"
N8N_PLATE_URL = "https://n8n.marshall321.org/webhook/agent/plate"

# =====================================================
# 🤖 MODEL
# =====================================================
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = "gpt-4o"

# =====================================================
# FASTAPI APP
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
# MEM0 (UNCHANGED)
# =====================================================
async def mem0_search(user_id: str, query: str):
    if not MEMO_API_KEY:
        return []
    headers = {"Authorization": f"Token MEMO_API_KEY"}
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
# NOTION PROMPT (UNCHANGED)
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
# n8n HELPERS (UNCHANGED)
# =====================================================
async def send_to_n8n(url: str, message: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(url, json={"message": message})
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
# NORMALIZATION HELPERS (UNCHANGED)
# =====================================================
def _normalize(m: str):
    m = m.lower().strip()
    m = "".join(ch for ch in m if ch not in string.punctuation)
    return " ".join(m.split())

def _is_similar(a: str, b: str):
    return bool(a and b and (a == b or a.startswith(b) or b.startswith(a) or a in b or b in a))

# =====================================================
# 🎤 WEBSOCKET HANDLER
# =====================================================
@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):

    await ws.accept()

    user_id = "solomon_roth"
    recent_msgs = []
    processed_messages = set()

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

    prompt = await get_notion_prompt()
    greet = prompt.splitlines()[0] if prompt else "Hello Solomon, I’m Silas."

    # GREETING TTS — UNCHANGED
    try:
        tts_greet = await openai_client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=greet
        )
        await ws.send_bytes(await tts_greet.aread())
    except Exception as e:
        log.error(f"❌ Greeting TTS error: {e}")

    # =====================================================
    # CONNECT TO DEEPGRAM (UPDATED)
    # =====================================================
    if not DEEPGRAM_API_KEY:
        log.error("❌ No DEEPGRAM_API_KEY set in environment.")
        return

    # ✅ FINAL NOVA-2 URL WITH TIMEOUT FIX
    dg_url = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-2"
        "&encoding=linear16"
        "&sample_rate=48000"
        "&no_audio_timeout_ms=60000"
    )

    try:
        dg_ws = await websockets.connect(
            dg_url,
            additional_headers=[("Authorization", f"Token {DEEPGRAM_API_KEY}")],
            ping_interval=None
        )
    except Exception as e:
        log.error(f"❌ Failed to connect to Deepgram WS: {e}")
        return

    # =====================================================
    # NEW PARALLEL DEEPGRAM LISTENER
    # =====================================================
    dg_queue = Queue()

    async def deepgram_listener_task():
        try:
            async for raw in dg_ws:
                try:
                    data = json.loads(raw)

                    if data.get("type") not in (
                        "Results",
                        "ResultCreated",
                        "ResultPartial",
                        "UtteranceEnd",
                        "UtteranceStarted",
                        "Metadata"
                    ):
                        continue

                    channel = data.get("channel", {})
                    alts = channel.get("alternatives", [])

                    transcript = ""
                    if alts:
                        transcript = alts[0].get("transcript", "").strip()

                    if transcript:
                        await dg_queue.put(transcript)

                except Exception as e:
                    log.error(f"❌ DG parse error: {e}")

        except Exception as e:
            log.error(f"❌ DG listener fatal: {e}")

    asyncio.create_task(deepgram_listener_task())

    # MAIN LOOP
    try:
        while True:

            try:
                data = await ws.receive()
            except RuntimeError:
                break
            except WebSocketDisconnect:
                break

            if data["type"] != "websocket.receive":
                continue

            if "bytes" not in data or data["bytes"] is None:
                continue

            audio_bytes = data["bytes"]

            # PCM ALIGNMENT FIX
            pcm = bytearray(audio_bytes)
            audio_bytes = bytes(pcm)

            # PCM SAMPLE LOGGING
            import struct
            try:
                samples = struct.unpack("<10h", audio_bytes[:20])
                log.info(f"PCM samples[0:10] = {list(samples)}")
            except Exception as e:
                log.error(f"sample unpack error: {e}")

            log.info(f"📡 PCM audio received — {len(audio_bytes)} bytes")

            # BACKEND PCM RMS + PEAK LOGGING
            try:
                if len(audio_bytes) >= 2:
                    total_samples = len(audio_bytes) // 2
                    all_samples = struct.unpack("<" + "h" * total_samples, audio_bytes[: total_samples * 2])
                    peak = max(abs(s) for s in all_samples)
                    rms = (sum(s * s for s in all_samples) / total_samples) ** 0.5
                    log.info(f"🔊 PCM STATS — RMS={rms:.2f}, Peak={peak}")
            except Exception as e:
                log.error(f"PCM stats error: {e}")

            # SEND AUDIO TO DEEPGRAM
            try:
                await dg_ws.send(audio_bytes)
            except Exception as e:
                log.error(f"❌ Error sending audio to Deepgram WS: {e}")
                continue

            # GET TRANSCRIPT FROM QUEUE
            transcript = ""
            try:
                transcript = await asyncio.wait_for(dg_queue.get(), timeout=1.0)
                log.info(f"📝 DG transcript: {transcript}")
            except asyncio.TimeoutError:
                continue

            if not transcript or len(transcript) < 3 or not any(ch.isalpha() for ch in transcript):
                continue

            msg = transcript
            norm = _normalize(msg)
            now = time.time()
            recent_msgs = [(m, t) for (m, t) in recent_msgs if now - t < 2]
            if any(_is_similar(m, norm) for (m, t) in recent_msgs):
                continue
            recent_msgs.append((norm, now))

            mems = await mem0_search(user_id, msg)
            ctx = memory_context(mems)
            sys_prompt = f"{prompt}\n\nFacts:\n{ctx}"
            lower = msg.lower()

            if any(k in lower for k in plate_kw):
                if msg in processed_messages:
                    continue
                processed_messages.add(msg)

                reply = await send_to_n8n(N8N_PLATE_URL, msg)

                try:
                    tts = await openai_client.audio.speech.create(
                        model="gpt-4o-mini-tts",
                        voice="alloy",
                        input=reply
                    )
                    await ws.send_bytes(await tts.aread())
                except Exception as e:
                    log.error(f"❌ TTS plate error: {e}")
                continue

            if any(k in lower for k in calendar_kw):
                reply = await send_to_n8n(N8N_CALENDAR_URL, msg)

                try:
                    tts = await openai_client.audio.speech.create(
                        model="gpt-4o-mini-tts",
                        voice="alloy",
                        input=reply
                    )
                    await ws.send_bytes(await tts.aread())
                except Exception as e:
                    log.error(f"❌ TTS calendar error: {e}")

                continue

            try:
                stream = await openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": msg},
                    ],
                    stream=True,
                )

                buffer = ""

                async for chunk in stream:
                    delta = getattr(chunk.choices[0].delta, "content", "")
                    if delta:
                        buffer += delta

                        if len(buffer) > 40:
                            try:
                                tts = await openai_client.audio.speech.create(
                                    model="gpt-4o-mini-tts",
                                    voice="alloy",
                                    input=buffer
                                )
                                await ws.send_bytes(await tts.aread())
                            except Exception as e:
                                log.error(f"❌ TTS stream-chunk error: {e}")
                            buffer = ""

                if buffer.strip():
                    try:
                        tts = await openai_client.audio.speech.create(
                            model="gpt-4o-mini-tts",
                            voice="alloy",
                            input=buffer
                        )
                        await ws.send_bytes(await tts.aread())
                    except Exception as e:
                        log.error(f"❌ TTS final-chunk error: {e}")

                asyncio.create_task(mem0_add(user_id, msg))

            except Exception as e:
                log.error(f"LLM error: {e}")

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await dg_ws.close()
        except:
            pass

# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

