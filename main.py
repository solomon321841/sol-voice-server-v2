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
USE_SSML = os.getenv("USE_SSML", "0") == "1"
CHUNK_CHAR_THRESHOLD = int(os.getenv("CHUNK_CHAR_THRESHOLD", "40"))
PUNCTUATE_WITH_LLM = os.getenv("PUNCTUATE_WITH_LLM", "0") == "1"
COGNITIVE_PACING_MS = int(os.getenv("COGNITIVE_PACING_MS", "60"))
SPEECH_RESHAPE = os.getenv("SPEECH_RESHAPE", "0") == "1"
MOMENTUM_ENABLED = os.getenv("MOMENTUM_ENABLED", "1") == "1"
BASE_PROSODY_RATE = float(os.getenv("BASE_PROSODY_RATE", "1.30"))
MIN_PROSODY_RATE = float(os.getenv("MIN_PROSODY_RATE", "1.20"))
MAX_PROSODY_RATE = float(os.getenv("MAX_PROSODY_RATE", "1.55"))
MOMENTUM_ALPHA = float(os.getenv("MOMENTUM_ALPHA", "0.15"))

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
_recent_rates = deque(maxlen=5)

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
        return "You are Solomon Roth's personal AI assistant, Silas."
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
            return "\n".join(parts).strip() or "You are Solomon Roth's AI assistant, Silas."
    except Exception as e:
        log.error(f"❌ Notion error: {e}")
        return "You are Solomon Roth's AI assistant, Silas."


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
# Utility: prepare TTS input (no SSML wrapper to avoid speaking "speak")
# =====================================================
def escape_for_ssml(s: str) -> str:
    return html.escape(s, quote=False)


async def reshape_for_speech(text: str) -> str:
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
# with Deepgram reconnect loop and exponential backoff
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
    greet = prompt.splitlines()[0] if prompt else "Hello Solomon, I'm Silas."
    system_msg_static = (
        prompt
        + "\n\nSpeaking style: Respond concisely in 1–3 sentences, like live conversation.  "
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

    # Queues & task tracking
    incoming_audio_queue: Queue = Queue()
    dg_transcript_queue: Queue = Queue()
    tts_tasks_by_turn: Dict[int, deque] = {}
    tts_locks_by_turn: Dict[int, asyncio.Lock] = {}
    MAX_TTS_TASKS_PER_TURN = 3
    last_audio_time = time.time()
    last_interrupt_ts = 0.0
    INTERRUPT_DEBOUNCE_SEC = 0.2

    # Flag to track client connection status
    client_connected = True

    # -----------------------------------------------------
    # Reader loop: single consumer of ws.receive() to handle both text and bytes
    # -----------------------------------------------------
    async def ws_reader():
        nonlocal last_audio_time, turn_id, current_active_turn_id, client_connected, last_interrupt_ts
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
                            now_ts = time.time()
                            if now_ts - last_interrupt_ts < INTERRUPT_DEBOUNCE_SEC:
                                log.info("⏳ Ignoring rapid interrupt (debounced)")
                                continue
                            last_interrupt_ts = now_ts
                            turn_id += 1
                            current_active_turn_id = turn_id
                            log.info(f"⏹️ Received interrupt from client — new active turn {current_active_turn_id}")
                            for t_id, tasks in list(tts_tasks_by_turn.items()):
                                if t_id != current_active_turn_id:
                                    for t in list(tasks):
                                        try:
                                            t.cancel()
                                        except Exception:
                                            pass
                                    tts_tasks_by_turn.pop(t_id, None)
                                    tts_locks_by_turn.pop(t_id, None)
                        elif typ == "final_silence":
                            await dg_transcript_queue.put({"final_silence": True})
                        else:
                            log.debug(f"WS text message (ignored): {data}")
                    elif "bytes" in msg and msg["bytes"] is not None:
                        audio_bytes = msg["bytes"]
                        await incoming_audio_queue.put(audio_bytes)
                        last_audio_time = time.time()
                    else:
                        pass
                elif mtype == "websocket.disconnect":
                    log.info("WS reader noticed disconnect")
                    client_connected = False
                    break
        except WebSocketDisconnect:
            log.info("WS reader disconnected")
            client_connected = False
        except Exception as e:
            log.error(f"ws_reader fatal: {e}")
            client_connected = False

    reader_task = asyncio.create_task(ws_reader())

    # -----------------------------------------------------
    # Helper: spawn a TTS generation-and-send task (non-blocking, serialized per turn)
    # -----------------------------------------------------
    async def _tts_and_send(tts_text: str, t_turn: int):
        try:
            if t_turn != current_active_turn_id:
                log.info(f"🔁 TTS task for turn {t_turn} cancelled before create (active={current_active_turn_id})")
                return

            lock = tts_locks_by_turn.setdefault(t_turn, asyncio.Lock())
            async with lock:
                if t_turn != current_active_turn_id:
                    log.info(f"🔁 TTS task for turn {t_turn} cancelled before generation (active={current_active_turn_id})")
                    return

                tts_payload = make_ssml_from_text(tts_text, None) if USE_SSML else tts_text
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

    def schedule_tts(tts_text: str, t_turn: int):
        queue = tts_tasks_by_turn.setdefault(t_turn, deque())
        while len(queue) >= MAX_TTS_TASKS_PER_TURN:
            oldest = queue.popleft()
            try:
                oldest.cancel()
            except Exception:
                pass
        task = asyncio.create_task(_tts_and_send(tts_text, t_turn))
        queue.append(task)

        def _cleanup(fut, t=t_turn):
            q = tts_tasks_by_turn.get(t)
            if q is not None:
                try:
                    q.remove(fut)
                except ValueError:
                    pass
                if not q:
                    tts_tasks_by_turn.pop(t, None)
                    tts_locks_by_turn.pop(t, None)

        task.add_done_callback(_cleanup)
        return task

    # -----------------------------------------------------
    # Transcript processor: delayed finalization with end-of-speech gate
    # -----------------------------------------------------
    async def transcript_processor():
        nonlocal recent_msgs, processed_messages, prompt, last_audio_time, turn_id, current_active_turn_id, chat_history
        pending_transcript = ""
        last_update_ts = 0.0
        buffered_user_text = ""
        last_final_ts = 0.0
        finalize_task: asyncio.Task = None
        FINAL_PAUSE_SEC = 0.3

        async def wait_and_finalize():
            try:
                await asyncio.sleep(FINAL_PAUSE_SEC)
                if time.time() - last_final_ts >= FINAL_PAUSE_SEC:
                    await commit_turn("timeout")
            except asyncio.CancelledError:
                return

        def schedule_finalize():
            nonlocal finalize_task
            if finalize_task and not finalize_task.done():
                finalize_task.cancel()
            finalize_task = asyncio.create_task(wait_and_finalize())

        async def commit_turn(reason: str):
            nonlocal buffered_user_text, finalize_task, turn_id, current_active_turn_id, chat_history
            if finalize_task and not finalize_task.done():
                finalize_task.cancel()
                finalize_task = None
            text = buffered_user_text.strip()
            if not text:
                return
            buffered_user_text = ""

            turn_id += 1
            current_turn = turn_id
            current_active_turn_id = current_turn
            chat_history.append({"role": "user", "content": text})
            if len(chat_history) > 8:
                chat_history[:] = chat_history[-8:]
            log.info(f"🎯 NEW TURN {current_turn}:  '{text}' (history len={len(chat_history)}) via {reason}")

            mems = await mem0_search(user_id, text)
            mem_facts = ""
            if mems:
                parts = []
                for m in mems[:3]:
                    txt = m.get("memory") or m.get("content") or m.get("text")
                    if txt:
                        parts.append(txt.strip())
                if parts:
                    mem_facts = "Memory: " + " ".join(parts)

            lower = text.lower()

            # Plate logic
            if any(k in lower for k in plate_kw):
                if text in processed_messages:
                    log.info(f"⏭ Plate msg already processed: '{text}'")
                    return
                processed_messages.add(text)
                reply = await send_to_n8n(N8N_PLATE_URL, text)
                if current_turn != current_active_turn_id:
                    log.info(f"🔁 Plate turn {current_turn} abandoned (active={current_active_turn_id})")
                    return
                schedule_tts(reply, current_turn)
                return

            # Calendar logic
            if any(k in lower for k in calendar_kw):
                reply = await send_to_n8n(N8N_CALENDAR_URL, text)
                if current_turn != current_active_turn_id:
                    log.info(f"🔁 Calendar turn {current_turn} abandoned (active={current_active_turn_id})")
                    return
                schedule_tts(reply, current_turn)
                return

            # General GPT streaming
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
                    if (
                        len(buffer) >= 70
                        or buffer.endswith((". ", "? ", "! ", ".", "?", "!"))
                    ):
                        chunk_text = buffer
                        buffer = ""
                        schedule_tts(chunk_text, current_turn)

                if buffer.strip() and current_turn == current_active_turn_id:
                    schedule_tts(buffer, current_turn)

                if assistant_full_text.strip() and current_turn == current_active_turn_id:
                    chat_history.append({"role": "assistant", "content": assistant_full_text.strip()})
                    if len(chat_history) > 8:
                        chat_history[:] = chat_history[-8:]
                    log.info(f"💾 Stored assistant turn {current_turn} in history (len={len(chat_history)})")

                asyncio.create_task(mem0_add(user_id, text))
            except Exception as e:
                log.error(f"LLM error: {e}")

        try:
            while True:
                transcript = await dg_transcript_queue.get()
                if transcript is None:
                    break

                if isinstance(transcript, dict) and transcript.get("final_silence"):
                    await commit_turn("final_silence")
                    continue

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

                buffered_user_text = (buffered_user_text + " " + msg).strip() if buffered_user_text else msg
                last_final_ts = time.time()
                schedule_finalize()
        except asyncio.CancelledError:
            log.info("transcript_processor cancelled")
        except Exception as e:
            log.error(f"❌ transcript_processor fatal: {e}")
        finally:
            if finalize_task and not finalize_task.done():
                finalize_task.cancel()

    transcript_task = asyncio.create_task(transcript_processor())

    # -----------------------------------------------------
    # Deepgram reconnect loop with exponential backoff
    # -----------------------------------------------------
    dg_url = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-2"
        "&encoding=linear16"
        "&sample_rate=48000"
    )

    # Backoff configuration
    INITIAL_BACKOFF = 0.5
    MAX_BACKOFF = 10.0
    current_backoff = INITIAL_BACKOFF

    async def clear_queue(q: Queue):
        """Drain all items from a queue without blocking."""
        while not q.empty():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def deepgram_connection_loop():
        nonlocal last_audio_time, client_connected, current_backoff

        while client_connected:
            dg_ws = None
            dg_listener_task = None
            dg_sender_task = None
            dg_keepalive_task = None

            try:
                # Attempt to connect to Deepgram
                log.info("🌐 Connecting to Deepgram...")
                try:
                    dg_ws = await websockets.connect(
                        dg_url,
                        additional_headers=[("Authorization", f"Token {DEEPGRAM_API_KEY}")],
                        ping_interval=None,
                        max_size=None,
                        close_timeout=0,
                    )
                    log.info("✅ Connected to Deepgram")
                    # Reset backoff on successful connection
                    current_backoff = INITIAL_BACKOFF
                except Exception as e:
                    log.error(f"❌ Failed to connect to Deepgram WS: {e}")
                    if not client_connected:
                        break
                    log.info(f"⏳ Retrying Deepgram connection in {current_backoff:.1f}s...")
                    await asyncio.sleep(current_backoff)
                    current_backoff = min(current_backoff * 2, MAX_BACKOFF)
                    continue

                # Clear queues to avoid stale data from previous connection
                await clear_queue(incoming_audio_queue)
                log.info("🧹 Cleared incoming_audio_queue for fresh connection")

                # -- Deepgram listener (accept only final transcripts)
                async def deepgram_listener_task_fn():
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
                                    alts = data.get("results", {}).get("alternatives", [])
                                transcript = ""
                                if alts and isinstance(alts, list):
                                    transcript = alts[0].get("transcript", "").strip()
                                if transcript:
                                    log.info(f"🧠 Deepgram final transcript: {transcript}")
                                    await dg_transcript_queue.put(transcript)
                            except Exception as e:
                                log.error(f"❌ DG parse error: {e}")
                    except websockets.exceptions.ConnectionClosed as e:
                        log.warning(f"⚠️ Deepgram connection closed: {e}")
                        raise
                    except Exception as e:
                        log.error(f"❌ DG listener fatal: {e}")
                        raise

                # -- DG audio sender: serializes sending bytes to Deepgram
                async def dg_audio_sender_fn():
                    try:
                        while True:
                            data = await incoming_audio_queue.get()
                            if data is None:
                                break
                            try:
                                if len(data) % 2 != 0:
                                    data = data + b"\x00"
                                await dg_ws.send(data)
                            except websockets.exceptions.ConnectionClosed as e:
                                log.warning(f"⚠️ DG sender: connection closed: {e}")
                                raise
                            except Exception as e:
                                log.error(f"❌ Error sending audio to Deepgram WS: {e}")
                                raise
                    except asyncio.CancelledError:
                        return
                    except Exception as e:
                        log.error(f"❌ DG sender fatal: {e}")
                        raise

                # -- DG keepalive task
                async def dg_keepalive_task_fn():
                    nonlocal last_audio_time
                    try:
                        while True:
                            await asyncio.sleep(1.2)
                            if time.time() - last_audio_time > 1.5:
                                try:
                                    silence = (b"\x00\x00") * 4800
                                    await dg_ws.send(silence)
                                    log.debug("📨 Sent DG keepalive silence")
                                except websockets.exceptions.ConnectionClosed as e:
                                    log.warning(f"⚠️ DG keepalive: connection closed: {e}")
                                    raise
                                except Exception as e:
                                    log.error(f"❌ Error sending keepalive to Deepgram: {e}")
                                    raise
                    except asyncio.CancelledError:
                        return
                    except Exception as e:
                        log.error(f"❌ DG keepalive fatal: {e}")
                        raise

                # Start all three DG tasks
                dg_listener_task = asyncio.create_task(deepgram_listener_task_fn())
                dg_sender_task = asyncio.create_task(dg_audio_sender_fn())
                dg_keepalive_task = asyncio.create_task(dg_keepalive_task_fn())

                dg_tasks = {dg_listener_task, dg_sender_task, dg_keepalive_task}

                # Wait for any DG task to complete (which indicates failure)
                # Also monitor reader_task to detect client disconnect
                done, pending = await asyncio.wait(
                    dg_tasks | {reader_task},
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Check if reader_task completed (client disconnected)
                if reader_task in done:
                    log.info("🔌 Client disconnected, stopping Deepgram reconnect loop")
                    client_connected = False
                    # Cancel remaining DG tasks
                    for task in pending:
                        if task != reader_task:
                            task.cancel()
                            try:
                                await task
                            except (asyncio.CancelledError, Exception):
                                pass
                    break

                # A DG task failed, need to reconnect
                failed_task = None
                for task in done:
                    if task in dg_tasks:
                        failed_task = task
                        break

                if failed_task:
                    try:
                        # This will raise the exception if the task failed
                        failed_task.result()
                    except Exception as e:
                        log.warning(f"⚠️ Deepgram task failed: {e}")

                log.info("🔄 Deepgram connection lost, initiating reconnect...")

            except Exception as e:
                log.error(f"❌ Unexpected error in Deepgram connection loop: {e}")

            finally:
                # Clean up DG tasks
                for task in [dg_listener_task, dg_sender_task, dg_keepalive_task]:
                    if task and not task.done():
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass

                # Close DG websocket
                if dg_ws:
                    try:
                        await dg_ws.close()
                        log.info("🔒 Closed Deepgram WebSocket")
                    except Exception:
                        pass

            # If client still connected, wait before reconnecting
            if client_connected:
                log.info(f"⏳ Waiting {current_backoff:.1f}s before Deepgram reconnect...")
                await asyncio.sleep(current_backoff)
                current_backoff = min(current_backoff * 2, MAX_BACKOFF)

    # Start the Deepgram connection loop
    dg_loop_task = asyncio.create_task(deepgram_connection_loop())

    # -----------------------------------------------------
    # Cleanup & shutdown handling
    # -----------------------------------------------------
    try:
        # Wait for the Deepgram loop to finish (which happens when client disconnects)
        await dg_loop_task
    except Exception as e:
        log.error(f"dg_loop_task error: {e}")
    finally:
        client_connected = False

        # Cancel tasks
        try:
            reader_task.cancel()
        except Exception:
            pass
        try:
            dg_loop_task.cancel()
        except Exception:
            pass
        try:
            transcript_task.cancel()
        except Exception:
            pass

        # Signal transcript processor to stop
        try:
            await dg_transcript_queue.put(None)
        except Exception:
            pass

        # Cancel outstanding TTS tasks
        for tasks in tts_tasks_by_turn.values():
            for t in list(tasks):
                try:
                    t.cancel()
                except Exception:
                    pass
        tts_tasks_by_turn.clear()
        tts_locks_by_turn.clear()

        # Wait for tasks to complete
        for task in [reader_task, transcript_task]:
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass

        log.info("🧹 WebSocket handler fully cleaned up")


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
