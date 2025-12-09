import os
import json
import logging
import asyncio
import time
import string
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
from collections import deque

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

# Feature flags / tuning via env
USE_SSML = os.getenv("USE_SSML", "0") == "1"  # default off to avoid speaking SSML tags
CHUNK_CHAR_THRESHOLD = int(os.getenv("CHUNK_CHAR_THRESHOLD", "40"))  # retained but unused
PUNCTUATE_WITH_LLM = os.getenv("PUNCTUATE_WITH_LLM", "0") == "1"  # retained but unused
COGNITIVE_PACING_MS = int(os.getenv("COGNITIVE_PACING_MS", "60"))  # retained but unused
SPEECH_RESHAPE = os.getenv("SPEECH_RESHAPE", "0") == "1"  # retained but unused
MOMENTUM_ENABLED = os.getenv("MOMENTUM_ENABLED", "1") == "1"  # retained but unused
BASE_PROSODY_RATE = float(os.getenv("BASE_PROSODY_RATE", "1.30"))  # retained but unused
MIN_PROSODY_RATE = float(os.getenv("MIN_PROSODY_RATE", "1.20"))  # retained but unused
MAX_PROSODY_RATE = float(os.getenv("MAX_PROSODY_RATE", "1.55"))  # retained but unused
MOMENTUM_ALPHA = float(os.getenv("MOMENTUM_ALPHA", "0.15"))  # retained but unused

# Finalization tuning (retained but unused)
FINAL_SILENCE_SEC = float(os.getenv("FINAL_SILENCE_SEC", "0.65"))
MIN_FINAL_WORDS = int(os.getenv("MIN_FINAL_WORDS", "3"))

# =====================================================
# n8n ENDPOINTS
# =====================================================
N8N_CALENDAR_URL = "https://n8n.marshall321.org/webhook/calendar-agent"
N8N_PLATE_URL = "https://n8n.marshall321.org/webhook/agent/plate"

# =====================================================
# MODEL
# =====================================================
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = "gpt-5-mini"
_recent_rates = deque(maxlen=5)  # retained but unused


# =====================================================
# FASTAPI
# =====================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():
    return {"status": "running", "message": "Silas backend is online."}


@app.get("/health")
async def health():
    return {"ok": True}


# =====================================================
# MEM0 HELPERS (from old main)
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
# NOTION PROMPT (from old main)
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
                    txt = "".join([r.get("plain_text", "") for r in blk["paragraph"]["rich_text"]])
                    parts.append(txt)
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
# Finalization helpers (retained but not used)
# =====================================================
def _looks_final(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t.endswith((".", "!", "?")):
        return True
    words = t.split()
    return len(words) >= MIN_FINAL_WORDS


# =====================================================
# Utility: prepare TTS input (no SSML wrapper to avoid speaking “speak”)
# =====================================================
def escape_for_ssml(s: str) -> str:  # basic escape for XML
    return html.escape(s, quote=False)


async def reshape_for_speech(text: str) -> str:
    # retained but unused in the low-latency path
    try:
        resp = await openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite the user's text into natural spoken English for voice TTS. Keep meaning, use contractions, avoid long sentences, avoid hedging, 1 short spoken clause. No markup.",
                },
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=min(60, max(20, len(text) // 2)),
        )
        out = resp.choices[0].message["content"].strip()
        return out or text
    except Exception:
        return text


def make_ssml_from_text(text: str, rate: float = None) -> str:
    t = text.strip()
    if not t:
        return t
    return t


# =====================================================
# WEBSOCKET HANDLER - improved: single receiver + cancellable TTS tasks
# =====================================================
@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):
    await ws.accept()
    user_id = "solomon_roth"
    recent_msgs = []
    processed_messages = set()
    chat_history: List[Dict] = []
    turn_id = 0
    current_active_turn_id = 0
    calendar_kw = ["calendar", "meeting", "schedule", "appointment"]
    plate_kw = ["plate", "add", "to-do", "task", "notion", "list"]

    prompt = await get_notion_prompt()
    greet = prompt.splitlines()[0] if prompt else "Hello Solomon, I’m Silas."
    system_msg_static = (
        prompt
        + "\n\nSpeaking style: Respond concisely in 1–3 sentences, like live conversation. "
        "Prioritize fast, direct answers over long explanations. "
        "Use conversational context from earlier turns to stay coherent, concise, and human-like."
    )

    # Greeting
    try:
        log.info("👋 Sending greeting TTS")
        greet_input = make_ssml_from_text(greet, None) if USE_SSML else greet
        tts_greet = await openai_client.audio.speech.create(model="gpt-4o-mini-tts", voice="cedar", input=greet_input)
        await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": 0}))
        await ws.send_bytes(await tts_greet.aread())
    except Exception as e:
        log.error(f"❌ Greeting TTS error: {e}")

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
        log.info("🌐 Connecting to Deepgram...")
        dg_ws = await websockets.connect(
            dg_url,
            additional_headers=[("Authorization", f"Token {DEEPGRAM_API_KEY}")],
            ping_interval=None,
            max_size=None,
            close_timeout=0,
        )
        log.info("✅ Connected to Deepgram")
    except Exception as e:
        log.error(f"❌ Failed to connect to Deepgram WS: {e}")
        return

    # Queues & task tracking
    incoming_audio_queue: Queue = Queue()  # bytes from client -> DG sender
    dg_transcript_queue: Queue = Queue()  # transcripts from DG listener -> transcript_processor
    tts_tasks_by_turn: Dict[int, Set[asyncio.Task]] = {}
    last_audio_time = time.time()

    # -- Deepgram listener (accept only final transcripts)
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
                    is_final = bool(data.get("is_final"))
                    if not is_final:
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
                        log.info(f"🧠 Deepgram final transcript: {transcript}")
                        await dg_transcript_queue.put(transcript)
                except Exception as e:
                    log.error(f"❌ DG parse error: {e}")
        except Exception as e:
            log.error(f"❌ DG listener fatal: {e}")

    asyncio.create_task(deepgram_listener_task())

    async def dg_keepalive_task():
        nonlocal last_audio_time
        try:
            while True:
                await asyncio.sleep(1.2)
                if time.time() - last_audio_time > 1.5:
                    try:
                        silence = (b"\x00\x00") * 4800
                        await dg_ws.send(silence)
                        log.info("📨 Sent DG keepalive silence")
                    except Exception as e:
                        log.error(f"❌ Error sending keepalive to Deepgram: {e}")
                        break
        except asyncio.CancelledError:
            return

    keepalive_task = asyncio.create_task(dg_keepalive_task())

    # -----------------------------------------------------
    # Reader loop: single consumer of ws.receive() to handle both text and bytes
    # -----------------------------------------------------
    async def ws_reader():
        nonlocal last_audio_time, turn_id, current_active_turn_id
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
                            # immediate interrupt: bump active turn and cancel outstanding TTS tasks
                            turn_id += 1
                            current_active_turn_id = turn_id
                            log.info(f"⏹️ Received interrupt from client — new active turn {current_active_turn_id}")
                            # cancel all outstanding tts tasks for older turns
                            for t_id, tasks in list(tts_tasks_by_turn.items()):
                                if t_id != current_active_turn_id:
                                    for t in tasks:
                                        try:
                                            t.cancel()
                                        except Exception:
                                            pass
                                    tts_tasks_by_turn.pop(t_id, None)
                        else:
                            # other text messages can be logged
                            log.debug(f"WS text message (ignored): {data}")
                    elif "bytes" in msg and msg["bytes"] is not None:
                        audio_bytes = msg["bytes"]
                        # audio_bytes are forwarded to DG sender via queue
                        await incoming_audio_queue.put(audio_bytes)
                        last_audio_time = time.time()
                    else:
                        # ignore other forms
                        pass
                elif mtype == "websocket.disconnect":
                    log.info("WS reader noticed disconnect")
                    break
        except WebSocketDisconnect:
            log.info("WS reader disconnected")
        except Exception as e:
            log.error(f"ws_reader fatal: {e}")

    reader_task = asyncio.create_task(ws_reader())

    # -----------------------------------------------------
    # DG audio sender: serializes sending bytes to Deepgram
    # -----------------------------------------------------
    async def dg_audio_sender():
        try:
            while True:
                data = await incoming_audio_queue.get()
                if data is None:
                    break
                try:
                    if len(data) % 2 != 0:
                        data = data + b"\x00"
                    await dg_ws.send(data)
                except Exception as e:
                    log.error(f"❌ Error sending audio to Deepgram WS: {e}")
        except asyncio.CancelledError:
            return

    dg_sender_task = asyncio.create_task(dg_audio_sender())

    # -----------------------------------------------------
    # Helper: spawn a TTS generation-and-send task (non-blocking)
    # Minimal path: no rate, no reshape, no punctuation, immediate send.
    # -----------------------------------------------------
    async def _tts_and_send(tts_text: str, t_turn: int):
        try:
            if USE_SSML:
                tts_payload = make_ssml_from_text(tts_text, None)
            else:
                tts_payload = tts_text

            if t_turn != current_active_turn_id:
                log.info(f"🔁 TTS task for turn {t_turn} cancelled before create (active={current_active_turn_id})")
                return

            tts = await openai_client.audio.speech.create(model="gpt-4o-mini-tts", voice="cedar", input=tts_payload)

            if t_turn != current_active_turn_id:
                log.info(f"🔁 TTS task for turn {t_turn} cancelled after generation (active={current_active_turn_id})")
                return

            try:
                await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": t_turn}))
            except Exception:
                pass

            audio_bytes = await tts.aread()
            if t_turn != current_active_turn_id:
                log.info(f"🔁 TTS task for turn {t_turn} cancelled after aread (active={current_active_turn_id})")
                return
            try:
                await ws.send_bytes(audio_bytes)
                log.info(f"🎙️ TTS SENT for turn={t_turn}, len={len(audio_bytes)}")
            except Exception as e:
                log.error(f"Failed to send TTS bytes for turn {t_turn}: {e}")
        except asyncio.CancelledError:
            log.info(f"🔁 TTS task for turn {t_turn} cancelled (task cancelled).")
            return
        except Exception as e:
            log.error(f"❌ TTS task error for turn {t_turn}: {e}")

    # -----------------------------------------------------
    # Transcript processor: immediate processing; no buffering/waits.
    # -----------------------------------------------------
    async def transcript_processor():
        nonlocal recent_msgs, processed_messages, prompt, last_audio_time, turn_id, current_active_turn_id, chat_history
        pending_transcript = ""
        last_update_ts = 0.0

        try:
            while True:
                transcript = await dg_transcript_queue.get()
                if transcript is None:
                    break

                if not transcript or len(transcript) < 1 or not any(ch.isalpha() for ch in transcript):
                    log.info("⏭ Ignoring very short / non-alpha transcript")
                    continue

                pending_transcript = transcript
                last_update_ts = time.time()

                msg = pending_transcript
                pending_transcript = ""

                norm = _normalize(msg)
                now = time.time()
                recent_msgs = [(m, t) for (m, t) in recent_msgs if now - t < 2]
                if any(_is_similar(m, norm) for (m, t) in recent_msgs):
                    log.info(f"⏭ Skipping near-duplicate transcript: '{msg}'")
                    continue
                recent_msgs.append((norm, now))

                # record user in history
                chat_history.append({"role": "user", "content": msg})

                # new turn
                turn_id += 1
                current_turn = turn_id
                current_active_turn_id = current_turn
                log.info(f"🎯 NEW TURN {current_turn}: '{msg}' (history len={len(chat_history)})")

                mems = await mem0_search(user_id, msg)
                mem_facts = ""
                if mems:
                    parts = []
                    for m in mems[:3]:
                        txt = m.get("memory") or m.get("content") or m.get("text")
                        if txt:
                            parts.append(txt.strip())
                    if parts:
                        mem_facts = "Memory: " + " ".join(parts)

                lower = msg.lower()

                # Plate logic
                if any(k in lower for k in plate_kw):
                    if msg in processed_messages:
                        log.info(f"⏭ Plate msg already processed: '{msg}'")
                        continue
                    processed_messages.add(msg)
                    reply = await send_to_n8n(N8N_PLATE_URL, msg)
                    if current_turn != current_active_turn_id:
                        log.info(f"🔁 Plate turn {current_turn} abandoned (active={current_active_turn_id})")
                        continue
                    t_task = asyncio.create_task(_tts_and_send(reply, current_turn))
                    tts_tasks_by_turn.setdefault(current_turn, set()).add(t_task)
                    t_task.add_done_callback(lambda fut, t=current_turn: tts_tasks_by_turn.get(t, set()).discard(fut))
                    continue

                # Calendar logic
                if any(k in lower for k in calendar_kw):
                    reply = await send_to_n8n(N8N_CALENDAR_URL, msg)
                    if current_turn != current_active_turn_id:
                        log.info(f"🔁 Calendar turn {current_turn} abandoned (active={current_active_turn_id})")
                        continue
                    t_task = asyncio.create_task(_tts_and_send(reply, current_turn))
                    tts_tasks_by_turn.setdefault(current_turn, set()).add(t_task)
                    t_task.add_done_callback(lambda fut, t=current_turn: tts_tasks_by_turn.get(t, set()).discard(fut))
                    continue

                # General GPT streaming -> spawn non-blocking TTS with micro-chunks
                try:
                    history_tail = chat_history[-3:]
                    messages = [{"role": "system", "content": system_msg_static}]
                    if mem_facts:
                        messages.append({"role": "system", "content": mem_facts})
                    messages += history_tail

                    log.info(f"🤖 GPT START turn={current_turn}, active={current_active_turn_id}, messages_len={len(messages)}")
                    stream = await openai_client.chat.completions.create(
                        model=GPT_MODEL,
                        messages=messages,
                        stream=True,
                    )

                    assistant_full_text = ""
                    buffer = ""

                    async for chunk in stream:
                        if current_turn != current_active_turn_id:
                            log.info(f"🔁 CANCEL STREAM turn={current_turn}, active={current_active_turn_id}")
                            break
                        delta = getattr(chunk.choices[0].delta, "content", "")
                        if not delta:
                            continue
                        assistant_full_text += delta
                        buffer += delta
                        if current_turn != current_active_turn_id:
                            log.info(f"🔁 Turn {current_turn} cancelled before TTS chunk.")
                            break
                        if len(buffer) >= 22 or buffer.endswith(('.', '?', '!', ',')):
                            chunk_text = buffer
                            buffer = ""
                            t_task = asyncio.create_task(_tts_and_send(chunk_text, current_turn))
                            tts_tasks_by_turn.setdefault(current_turn, set()).add(t_task)
                            t_task.add_done_callback(lambda fut, t=current_turn: tts_tasks_by_turn.get(t, set()).discard(fut))

                    if buffer.strip() and current_turn == current_active_turn_id:
                        t_task = asyncio.create_task(_tts_and_send(buffer, current_turn))
                        tts_tasks_by_turn.setdefault(current_turn, set()).add(t_task)
                        t_task.add_done_callback(lambda fut, t=current_turn: tts_tasks_by_turn.get(t, set()).discard(fut))

                    if assistant_full_text.strip() and current_turn == current_active_turn_id:
                        chat_history.append({"role": "assistant", "content": assistant_full_text.strip()})
                        log.info(f"💾 Stored assistant turn {current_turn} in history (len={len(chat_history)})")

                    asyncio.create_task(mem0_add(user_id, msg))
                except Exception as e:
                    log.error(f"LLM error: {e}")
        except Exception as e:
            log.error(f"❌ transcript_processor fatal: {e}")

    transcript_task = asyncio.create_task(transcript_processor())

    # -----------------------------------------------------
    # Cleanup & shutdown handling
    # -----------------------------------------------------
    try:
        # wait for reader to finish (client disconnect) — other tasks run independently
        await reader_task
    except Exception:
        pass
    finally:
        # cancel tasks/close connections
        try:
            reader_task.cancel()
        except Exception:
            pass
        try:
            dg_sender_task.cancel()
        except Exception:
            pass
        try:
            transcript_task.cancel()
        except Exception:
            pass
        try:
            keepalive_task.cancel()
        except Exception:
            pass
        # cancel outstanding TTS tasks
        for tasks in tts_tasks_by_turn.values():
            for t in tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
        try:
            await dg_ws.close()
        except Exception:
            pass


# =====================================================
# helper for n8n calls (aligned with old main behavior)
# =====================================================
async def send_to_n8n(url: str, text: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            payload = {"message": text}
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
                except Exception:
                    return r.text.strip()
            return "Sorry, the automation returned an unexpected response."
    except Exception as e:
        log.error(f"n8n error: {e}")
        return "Sorry, couldn't reach automation."


# =====================================================
# SERVER START
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
