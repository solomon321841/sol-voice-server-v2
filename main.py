]# main.py
# Improved TTS prosody and server-side concatenation of TTS chunks to remove gaps
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
import struct

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

USE_SSML = os.getenv("USE_SSML", "1") == "1"
CHUNK_CHAR_THRESHOLD = int(os.getenv("CHUNK_CHAR_THRESHOLD", "40"))
PUNCTUATE_WITH_LLM = os.getenv("PUNCTUATE_WITH_LLM", "0") == "1"

# Aggregation window in milliseconds: wait up to this long for more tts chunks to arrive
AGGREGATION_WINDOW_MS = int(os.getenv("AGGREGATION_WINDOW_MS", "120"))

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
# helpers (mem0, notion, normalization) — unchanged logic
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

def _normalize(m: str):
    m = m.lower().strip()
    m = "".join(ch for ch in m if ch not in string.punctuation)
    return " ".join(m.split())

def _is_similar(a: str, b: str):
    return bool(a and b and (a == b or a.startswith(b) or b.startswith(a) or a in b or b in a))

# =====================================================
# SSML / prosody helper — now only adds sentence breaks, no comma breaks
# Also slightly reduce speaking rate to make voice more natural
# =====================================================
def ensure_sentence_punctuation(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s[-1] not in ".!?":
        s = s + "."
    return s

def escape_for_ssml(s: str) -> str:
    return html.escape(s, quote=False)

def make_ssml_from_text(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    t = ensure_sentence_punctuation(t)
    t_esc = escape_for_ssml(t)
    # Only add breaks after sentence-ending punctuation to avoid chopping mid-sentence
    t_esc = t_esc.replace(". ", ".<break time=\"200ms\"/> ")
    # Slight prosody: set a slightly slower speaking rate; tune rate as desired
    ssml = f"<speak><prosody rate=\"0.98\">{t_esc}</prosody></speak>"
    return ssml

# =====================================================
# WAV concatenation helper
# This function expects inputs to be standard WAV RIFF files with identical format
# It extracts the 'fmt ' chunk and 'data' bytes and concatenates PCM bytes, producing a new WAV.
# =====================================================
def concat_wav_bytes(wav_bytes_list: List[bytes]) -> bytes:
    if not wav_bytes_list:
        return b''

    # Simple parser: find first 'fmt ' and 'data' chunks from first wav, reuse fmt for header.
    def parse_wav(b: bytes):
        # Verify RIFF header
        if len(b) < 44 or b[0:4] != b'RIFF' or b[8:12] != b'WAVE':
            raise ValueError("Not a WAV")
        idx = 12
        fmt_chunk = None
        data_chunk = None
        while idx + 8 <= len(b):
            chunk_id = b[idx:idx+4]
            chunk_size = int.from_bytes(b[idx+4:idx+8], 'little')
            chunk_data_start = idx + 8
            chunk_data_end = chunk_data_start + chunk_size
            if chunk_id == b'fmt ':
                fmt_chunk = b[idx:chunk_data_end]
            elif chunk_id == b'data':
                data_chunk = b[chunk_data_start:chunk_data_end]
            idx = chunk_data_end
            # handle pad byte if chunk_size odd
            if chunk_size % 2 == 1:
                idx += 1
        return fmt_chunk, data_chunk

    first_fmt, first_data = parse_wav(wav_bytes_list[0])
    if first_fmt is None or first_data is None:
        # fallback: return first bytes
        return wav_bytes_list[0]

    pcm_datas = [first_data]
    for b in wav_bytes_list[1:]:
        try:
            _, data = parse_wav(b)
            if data is None:
                continue
            pcm_datas.append(data)
        except Exception:
            # If parsing fails for some chunk, skip it
            continue

    total_data = b''.join(pcm_datas)
    # Build new RIFF header: take fmt from first wav and create new data chunk
    # Compute sizes
    fmt_chunk = first_fmt
    fmt_size = len(fmt_chunk) - 8  # chunk header excluded
    data_size = len(total_data)
    # RIFF chunk size = 4 (WAVE) + (8 + fmt_size) + (8 + data_size)
    riff_size = 4 + (8 + fmt_size) + (8 + data_size)
    out = bytearray()
    out += b'RIFF'
    out += (riff_size).to_bytes(4, 'little')
    out += b'WAVE'
    out += fmt_chunk
    out += b'data'
    out += (data_size).to_bytes(4, 'little')
    out += total_data
    return bytes(out)

# =====================================================
# WEBSOCKET handler: single-reader + per-turn TTS aggregation & cancellation
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

    # greeting
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
            close_timeout=0
        )
        log.info("✅ Connected to Deepgram")
    except Exception as e:
        log.error(f"❌ Failed to connect to Deepgram WS: {e}")
        return

    incoming_audio_queue: Queue = Queue()
    dg_transcript_queue: Queue = Queue()

    # Per-turn result queues and tracking of sender tasks
    tts_result_queues: Dict[int, asyncio.Queue] = {}
    tts_sender_tasks: Dict[int, asyncio.Task] = {}
    tts_tasks_by_turn: Dict[int, Set[asyncio.Task]] = {}

    last_audio_time = time.time()

    # Deepgram listener -> push transcripts into dg_transcript_queue
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
                        log.info(f"🧠 Deepgram partial/final transcript: {transcript}")
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

    # single ws reader handling both text and bytes
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
                            turn_id += 1
                            current_active_turn_id = turn_id
                            log.info(f"⏹️ Received interrupt from client — new active turn {current_active_turn_id}")
                            # cancel outstanding tts tasks and sender tasks for old turns
                            for t_id, tasks in list(tts_tasks_by_turn.items()):
                                if t_id != current_active_turn_id:
                                    for t in tasks:
                                        try:
                                            t.cancel()
                                        except Exception:
                                            pass
                                    tts_tasks_by_turn.pop(t_id, None)
                            for t_id, task in list(tts_sender_tasks.items()):
                                if t_id != current_active_turn_id:
                                    try:
                                        task.cancel()
                                    except Exception:
                                        pass
                                    tts_sender_tasks.pop(t_id, None)
                                    q = tts_result_queues.pop(t_id, None)
                                    if q:
                                        # drain queue
                                        try:
                                            while not q.empty():
                                                q.get_nowait()
                                        except Exception:
                                            pass
                        else:
                            log.debug(f"WS text message (ignored): {data}")
                    elif "bytes" in msg and msg["bytes"] is not None:
                        audio_bytes = msg["bytes"]
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

    # send audio to deepgram
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

    # TTS generator: create audio and put bytes into per-turn result queue (do NOT send immediately)
    async def _tts_generate_put(tts_text: str, t_turn: int):
        try:
            # build payload using SSML; avoid inserting commas
            tts_payload = make_ssml_from_text(tts_text) if USE_SSML else tts_text

            if t_turn != current_active_turn_id:
                log.info(f"🔁 TTS generation aborted pre-create for turn {t_turn}")
                return

            tts = await openai_client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=tts_payload
            )

            audio_bytes = await tts.aread()

            if t_turn != current_active_turn_id:
                log.info(f"🔁 TTS generation aborted post-create for turn {t_turn}")
                return

            # ensure queue exists for this turn
            q = tts_result_queues.get(t_turn)
            if not q:
                q = asyncio.Queue()
                tts_result_queues[t_turn] = q
            await q.put(audio_bytes)
            log.info(f"🔊 Generated TTS for turn {t_turn}, queued bytes len={len(audio_bytes)}")
        except asyncio.CancelledError:
            log.info(f"🔁 TTS generate task cancelled for turn {t_turn}")
            return
        except Exception as e:
            log.error(f"❌ _tts_generate_put error for turn {t_turn}: {e}")

    # per-turn sender: collect queued wav bytes within AGGREGATION_WINDOW_MS and send as a single concatenated WAV
    async def _tts_sender_for_turn(t_turn: int):
        q = tts_result_queues.get(t_turn)
        if q is None:
            return
        try:
            while True:
                try:
                    # wait for first item
                    first = await asyncio.wait_for(q.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    # nothing to send for a while; finish
                    break
                buffers = [first]
                # collect others that arrive shortly
                start = time.time()
                while (time.time() - start) * 1000.0 < AGGREGATION_WINDOW_MS:
                    try:
                        item = q.get_nowait()
                        buffers.append(item)
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.02)
                # concatenate WAVs safely and send
                try:
                    merged = concat_wav_bytes(buffers)
                except Exception as e:
                    log.debug(f"WAV concat failed, sending first buffer only: {e}")
                    merged = buffers[0]
                # notify client and send merged bytes
                try:
                    await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": t_turn}))
                except Exception:
                    pass
                try:
                    await ws.send_bytes(merged)
                    log.info(f"🎙️ Sent aggregated TTS for turn {t_turn}, merged_len={len(merged)}, parts={len(buffers)}")
                except Exception as e:
                    log.error(f"Failed to send aggregated TTS for turn {t_turn}: {e}")
                # if the queue is empty and no further items, exit loop (sender will restart if new items come and a sender is re-created)
                if q.empty():
                    # small grace sleep to group potential immediate follow-ups
                    await asyncio.sleep(0.02)
                    if q.empty():
                        break
        except asyncio.CancelledError:
            log.info(f"🔁 TTS sender cancelled for turn {t_turn}")
            return
        finally:
            tts_result_queues.pop(t_turn, None)
            tts_sender_tasks.pop(t_turn, None)

    # transcript processor: spawn tts generation tasks (that put bytes into per-turn queues)
    async def transcript_processor():
        nonlocal recent_msgs, processed_messages, prompt, last_audio_time, turn_id, current_active_turn_id, chat_history
        try:
            while True:
                transcript = await dg_transcript_queue.get()
                if transcript is None:
                    break
                if not transcript or len(transcript) < 3 or not any(ch.isalpha() for ch in transcript):
                    continue
                msg = transcript
                norm = _normalize(msg)
                now = time.time()
                recent_msgs = [(m, t) for (m, t) in recent_msgs if now - t < 2]
                if any(_is_similar(m, norm) for (m, t) in recent_msgs):
                    continue
                recent_msgs.append((norm, now))

                chat_history.append({"role": "user", "content": msg})

                turn_id += 1
                current_turn = turn_id
                current_active_turn_id = current_turn
                log.info(f"🎯 NEW TURN {current_turn}: '{msg}' (history len={len(chat_history)})")

                mems = await mem0_search(user_id, msg)
                ctx = memory_context(mems)
                sys_prompt = f"{prompt}\n\nFacts:\n{ctx}"
                system_msg = (
                    sys_prompt
                    + "\n\nSpeaking style: Respond concisely in 1–3 sentences, like live conversation. "
                      "Prioritize fast, direct answers over long explanations."
                )

                lower = msg.lower()

                # plate/calendar logic (same pattern)
                if any(k in lower for k in plate_kw):
                    if msg in processed_messages:
                        continue
                    processed_messages.add(msg)
                    reply = await send_to_n8n(N8N_PLATE_URL, msg)
                    if current_turn != current_active_turn_id:
                        continue
                    # spawn generator + sender for this turn
                    gen_task = asyncio.create_task(_tts_generate_put(reply, current_turn))
                    tts_tasks_by_turn.setdefault(current_turn, set()).add(gen_task)
                    gen_task.add_done_callback(lambda fut, t=current_turn: tts_tasks_by_turn.get(t, set()).discard(fut))
                    # ensure a sender exists for this turn
                    if current_turn not in tts_sender_tasks:
                        tts_sender_tasks[current_turn] = asyncio.create_task(_tts_sender_for_turn(current_turn))
                    continue

                if any(k in lower for k in calendar_kw):
                    reply = await send_to_n8n(N8N_CALENDAR_URL, msg)
                    if current_turn != current_active_turn_id:
                        continue
                    gen_task = asyncio.create_task(_tts_generate_put(reply, current_turn))
                    tts_tasks_by_turn.setdefault(current_turn, set()).add(gen_task)
                    gen_task.add_done_callback(lambda fut, t=current_turn: tts_tasks_by_turn.get(t, set()).discard(fut))
                    if current_turn not in tts_sender_tasks:
                        tts_sender_tasks[current_turn] = asyncio.create_task(_tts_sender_for_turn(current_turn))
                    continue

                # General GPT streaming: spawn TTS generate tasks for chunks (non-blocking)
                try:
                    messages = [{"role": "system", "content": system_msg}] + chat_history
                    log.info(f"🤖 GPT START turn={current_turn}, active={current_active_turn_id}, messages_len={len(messages)}")

                    stream = await openai_client.chat.completions.create(
                        model=GPT_MODEL,
                        messages=messages,
                        stream=True,
                    )

                    buffer = ""
                    assistant_full_text = ""

                    async for chunk in stream:
                        if current_turn != current_active_turn_id:
                            log.info(f"🔁 CANCEL STREAM turn={current_turn}, active={current_active_turn_id}")
                            break

                        delta = getattr(chunk.choices[0].delta, "content", "")
                        if not delta:
                            continue

                        assistant_full_text += delta
                        buffer += delta

                        if len(buffer) >= CHUNK_CHAR_THRESHOLD:
                            if current_turn != current_active_turn_id:
                                break

                            chunk_text = buffer
                            buffer = ""
                            gen_task = asyncio.create_task(_tts_generate_put(chunk_text, current_turn))
                            tts_tasks_by_turn.setdefault(current_turn, set()).add(gen_task)
                            gen_task.add_done_callback(lambda fut, t=current_turn: tts_tasks_by_turn.get(t, set()).discard(fut))
                            # ensure sender exists
                            if current_turn not in tts_sender_tasks:
                                tts_sender_tasks[current_turn] = asyncio.create_task(_tts_sender_for_turn(current_turn))

                    # final buffer -> generate final tts
                    if buffer.strip() and current_turn == current_active_turn_id:
                        gen_task = asyncio.create_task(_tts_generate_put(buffer, current_turn))
                        tts_tasks_by_turn.setdefault(current_turn, set()).add(gen_task)
                        gen_task.add_done_callback(lambda fut, t=current_turn: tts_tasks_by_turn.get(t, set()).discard(fut))
                        if current_turn not in tts_sender_tasks:
                            tts_sender_tasks[current_turn] = asyncio.create_task(_tts_sender_for_turn(current_turn))

                    if assistant_full_text.strip() and current_turn == current_active_turn_id:
                        chat_history.append({"role": "assistant", "content": assistant_full_text.strip()})
                        log.info(f"💾 Stored assistant turn {current_turn} in history (len={len(chat_history)})")

                    asyncio.create_task(mem0_add(user_id, msg))

                except Exception as e:
                    log.error(f"LLM error: {e}")

        except Exception as e:
            log.error(f"❌ transcript_processor fatal: {e}")

    transcript_task = asyncio.create_task(transcript_processor())

    # wait for reader to finish (client disconnect) and cleanup
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
        # cancel outstanding TTS generator tasks
        for tasks in tts_tasks_by_turn.values():
            for t in tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
        # cancel sender tasks
        for t in list(tts_sender_tasks.values()):
            try:
                t.cancel()
            except Exception:
                pass
        try:
            await dg_ws.close()
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
# server start
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
