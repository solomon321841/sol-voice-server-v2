# (Full file with a minimal, focused change: decouple audio forwarding from transcript processing.
# The previous implementation awaited dg_queue.get() inside the audio-forwarding loop which blocked
# reading further audio from the browser and caused Deepgram timeouts (1011). This version adds a
# dedicated transcript_processor task that consumes transcripts from dg_queue asynchronously so
# the audio-forwarding loop can continuously read and forward binary audio frames to Deepgram.)
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
from asyncio import Queue   # ADDED earlier, unchanged

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
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()

# =====================================================
# n8n ENDPOINTS
# =====================================================
N8N_CALENDAR_URL = "https://n8n.marshall321.org/webhook/calendar-agent"
N8N_PLATE_URL = "https://n8n.marshall321.org/webhook/agent/plate"

# =====================================================
# MODEL
# =====================================================
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = "gpt-4o"

# How big each TTS text chunk should be (characters) for natural speech
CHUNK_CHAR_THRESHOLD = 90  # tune between ~60–120

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

# ... (mem0, notion, n8n helpers unchanged) ...

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

# ... (get_notion_prompt and n8n helpers unchanged) ...

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
# NORMALIZATION — UNCHANGED
# =====================================================
def _normalize(m: str):
    m = m.lower().strip()
    m = "".join(ch for ch in m if ch not in string.punctuation)
    return " ".join(m.split())

def _is_similar(a: str, b: str):
    return bool(a and b and (a == b or a.startswith(b) or b.startswith(a) or a in b or b in a))

# =====================================================
# WEBSOCKET HANDLER
# =====================================================
@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):

    await ws.accept()

    user_id = "solomon_roth"
    recent_msgs = []
    processed_messages = set()

    # TURN COUNTER FOR THIS CONNECTION
    # Each accepted transcript increments turn_id.
    turn_id = 0

    # This tracks which turn is currently allowed to generate TTS.
    # When a new transcript arrives, this is set to that turn, and older
    # turns stop generating further TTS chunks.
    current_active_turn_id = 0

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

    # GREETING TTS (turn_id 0, not part of interruption logic)
    try:
        tts_greet = await openai_client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=greet
        )
        await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": 0}))
        await ws.send_bytes(await tts_greet.aread())
    except Exception as e:
        log.error(f"❌ Greeting TTS error: {e}")

    # =====================================================
    # DEEPGRAM CONNECTION
    # =====================================================
    if not DEEPGRAM_API_KEY:
        log.error("❌ No DEEPGRAM_API_KEY set.")
        return

    dg_url = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-2"
        "&encoding=linear16"
        "&sample_rate=48000"
    )

    try:
        dg_ws = await websockets.connect(
            dg_url,
            additional_headers=[("Authorization", f"Token {DEEPGRAM_API_KEY}")],
            ping_interval=None,
            max_size=None,
            close_timeout=0
        )
    except Exception as e:
        log.error(f"❌ Failed to connect to Deepgram WS: {e}")
        return

    # =====================================================
    # PARALLEL DEEPGRAM LISTENER
    # =====================================================
    dg_queue = Queue()

    async def deepgram_listener_task():
        try:
            async for raw in dg_ws:
                try:
                    if isinstance(raw, (bytes, bytearray)):
                        raw_text = raw.decode("utf-8", errors="ignore")
                    else:
                        raw_text = raw

                    data = json.loads(raw_text)
                    if not isinstance(data, dict):
                        continue

                    alts = []
                    if "channel" in data and isinstance(data["channel"], dict):
                        alts = data["channel"].get("alternatives", [])
                    elif "results" in data and isinstance(data["results"], dict):
                        ch = data["results"].get("channels", [])
                        if ch and isinstance(ch, list):
                            alts = ch[0].get("alternatives", [])
                        else:
                            alts = data["results"].get("alternatives", [])

                    transcript = ""
                    if alts and isinstance(alts, list):
                        transcript = alts[0].get("transcript", "").strip()

                    if transcript:
                        await dg_queue.put(transcript)

                except Exception as e:
                    log.error(f"❌ DG parse error: {e}")
        except Exception as e:
            log.error(f"❌ DG listener fatal: {e}")

    asyncio.create_task(deepgram_listener_task())

    # Keep last audio timestamp for backend keepalive
    last_audio_time = time.time()

    async def dg_keepalive_task():
        nonlocal last_audio_time
        try:
            while True:
                await asyncio.sleep(1.2)
                if time.time() - last_audio_time > 1.5:
                    try:
                        silence = (b'\x00\x00') * 4800  # 100ms @ 48kHz
                        await dg_ws.send(silence)
                    except Exception as e:
                        log.error(f"❌ Error sending keepalive to Deepgram: {e}")
                        break
        except asyncio.CancelledError:
            return

    keepalive_task = asyncio.create_task(dg_keepalive_task())

    # =====================================================
    # Transcript processor — TURN TAGGING + CANCELLATION
    # =====================================================
    async def transcript_processor():
        nonlocal recent_msgs, processed_messages, prompt, last_audio_time, turn_id, current_active_turn_id
        try:
            while True:
                try:
                    transcript = await dg_queue.get()
                except asyncio.CancelledError:
                    break

                if not transcript:
                    continue

                log.info(f"📝 DG transcript: {transcript}")

                if not transcript or len(transcript) < 3 or not any(ch.isalpha() for ch in transcript):
                    continue

                msg = transcript
                norm = _normalize(msg)
                now = time.time()
                recent_msgs = [(m, t) for (m, t) in recent_msgs if now - t < 2]
                if any(_is_similar(m, norm) for (m, t) in recent_msgs):
                    continue
                recent_msgs.append((norm, now))

                # NEW TURN: each accepted transcript is a new turn.
                turn_id += 1
                current_turn = turn_id

                # IMPORTANT: Mark this turn as the ONLY active one.
                # Any older GPT loops still running will see mismatch and stop.
                current_active_turn_id = current_turn

                # memory + notion context
                mems = await mem0_search(user_id, msg)
                ctx = memory_context(mems)
                sys_prompt = f"{prompt}\n\nFacts:\n{ctx}"
                system_msg = (
                    sys_prompt
                    + "\n\nSpeaking style: Respond concisely in 1–3 sentences, like live conversation. "
                      "Prioritize fast, direct answers over long explanations."
                )

                lower = msg.lower()

                # Plate logic
                if any(k in lower for k in plate_kw):
                    if msg in processed_messages:
                        continue
                    processed_messages.add(msg)

                    reply = await send_to_n8n(N8N_PLATE_URL, msg)

                    # Before doing TTS, check if this turn is still active
                    if current_turn != current_active_turn_id:
                        continue

                    try:
                        tts = await openai_client.audio.speech.create(
                            model="gpt-4o-mini-tts",
                            voice="alloy",
                            input=reply
                        )
                        # Check again after TTS call (in case of quick double-interrupt)
                        if current_turn != current_active_turn_id:
                            continue
                        try:
                            await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                        except Exception:
                            pass
                        await ws.send_bytes(await tts.aread())
                    except Exception as e:
                        log.error(f"❌ TTS plate error: {e}")
                    continue

                # Calendar logic
                if any(k in lower for k in calendar_kw):
                    reply = await send_to_n8n(N8N_CALENDAR_URL, msg)

                    if current_turn != current_active_turn_id:
                        continue

                    try:
                        tts = await openai_client.audio.speech.create(
                            model="gpt-4o-mini-tts",
                            voice="alloy",
                            input=reply
                        )
                        if current_turn != current_active_turn_id:
                            continue
                        try:
                            await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                        except Exception:
                            pass
                        await ws.send_bytes(await tts.aread())
                    except Exception as e:
                        log.error(f"❌ TTS calendar error: {e}")
                    continue

                # General GPT logic
                try:
                    stream = await openai_client.chat.completions.create(
                        model=GPT_MODEL,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": msg},
                        ],
                        stream=True,
                    )

                    buffer = ""

                    async for chunk in stream:
                        # If a newer turn became active while streaming, stop immediately.
                        if current_turn != current_active_turn_id:
                            log.info(f"🔁 Turn {current_turn} cancelled mid-stream (newer turn {current_active_turn_id} active).")
                            break

                        delta = getattr(chunk.choices[0].delta, "content", "")
                        if not delta:
                            continue

                        buffer += delta

                        if len(buffer) > CHUNK_CHAR_THRESHOLD:
                            # Double-check cancellation before doing TTS
                            if current_turn != current_active_turn_id:
                                log.info(f"🔁 Turn {current_turn} cancelled before TTS chunk.")
                                break

                            try:
                                tts = await openai_client.audio.speech.create(
                                    model="gpt-4o-mini-tts",
                                    voice="alloy",
                                    input=buffer
                                )
                                # Check again after TTS call
                                if current_turn != current_active_turn_id:
                                    log.info(f"🔁 Turn {current_turn} cancelled after TTS chunk generation.")
                                    break
                                try:
                                    await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                                except Exception:
                                    pass
                                await ws.send_bytes(await tts.aread())
                            except Exception as e:
                                log.error(f"❌ TTS stream-chunk error: {e}")
                            buffer = ""

                    # Final buffer (if any) for this turn
                    if buffer.strip() and current_turn == current_active_turn_id:
                        try:
                            tts = await openai_client.audio.speech.create(
                                model="gpt-4o-mini-tts",
                                voice="alloy",
                                input=buffer
                            )
                            if current_turn == current_active_turn_id:
                                try:
                                    await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                                except Exception:
                                    pass
                                await ws.send_bytes(await tts.aread())
                        except Exception as e:
                            log.error(f"❌ TTS final-chunk error: {e}")

                    asyncio.create_task(mem0_add(user_id, msg))

                except Exception as e:
                    log.error(f"LLM error: {e}")

        except Exception as e:
            log.error(f"❌ transcript_processor fatal: {e}")

    transcript_task = asyncio.create_task(transcript_processor())

    # =====================================================
    # MAIN LOOP — receive raw bytes from browser and forward to Deepgram
    # =====================================================
    try:
        while True:
            try:
                audio_bytes = await ws.receive_bytes()
            except WebSocketDisconnect:
                log.info("Browser websocket disconnected")
                break
            except Exception as e:
                log.error(f"WebSocket receive error: {e}")
                await asyncio.sleep(0.05)
                continue

            if not audio_bytes:
                continue

            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes + b'\x00'

            last_audio_time = time.time()

            import struct
            try:
                if len(audio_bytes) >= 20:
                    samples = struct.unpack("<10h", audio_bytes[:20])
                    log.info(f"PCM samples[0:10] = {list(samples)}")
            except Exception as e:
                log.error(f"sample unpack error: {e}")

            log.info(f"📡 PCM audio received — {len(audio_bytes)} bytes")

            try:
                if len(audio_bytes) >= 2:
                    total_samples = len(audio_bytes) // 2
                    all_samples = struct.unpack("<" + "h" * total_samples, audio_bytes[: total_samples * 2])
                    peak = max(abs(s) for s in all_samples) if all_samples else 0
                    rms = (sum(s * s for s in all_samples) / total_samples) ** 0.5 if total_samples > 0 else 0
                    log.info(f"🔊 PCM STATS — RMS={rms:.2f}, Peak={peak}")
            except Exception as e:
                log.error(f"PCM stats error: {e}")

            try:
                await dg_ws.send(audio_bytes)
            except Exception as e:
                log.error(f"❌ Error sending audio to Deepgram WS: {e}")
                continue

    except WebSocketDisconnect:
        pass
    finally:
        try:
            keepalive_task.cancel()
        except:
            pass
        try:
            transcript_task.cancel()
        except:
            pass
        try:
            await dg_ws.close()
        except:
            pass

# =====================================================
# SERVER START
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
