# main.py
# Minimal changes on top of your v5 server to produce more natural TTS:
# - Optional "natural" mode: wait for full assistant text, optionally punctuate with LLM, then call TTS once
# - Improved SSML prosody (no mid-sentence comma breaks — only sentence pauses)
# - Configurable with env NATURAL_SPEECH and PUNCTUATE_WITH_LLM
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
import re

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

# Behavior tuning
USE_SSML = os.getenv("USE_SSML", "1") == "1"
PUNCTUATE_WITH_LLM = os.getenv("PUNCTUATE_WITH_LLM", "1") == "1"
NATURAL_SPEECH = os.getenv("NATURAL_SPEECH", "1") == "1"  # if 1 => single TTS for full assistant response (more natural)
CHUNK_CHAR_THRESHOLD = int(os.getenv("CHUNK_CHAR_THRESHOLD", "40"))  # still used when NATURAL_SPEECH=0
# If NATURAL_SPEECH=1 you can increase this to reduce bandwidth for streaming tokens but it won't affect TTS.

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
# helpers (mem0, notion, normalization) — keep existing behavior
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
# SSML & punctuation helpers
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
    # Only add pauses at sentence boundaries and mild prosody adjustments
    t = text.strip()
    if not t:
        return t
    # Ensure sentence endings for SSML pauses
    t = ensure_sentence_punctuation(t)
    t_esc = escape_for_ssml(t)
    # Only sentence pauses (no comma pauses to avoid mid-sentence chopping)
    t_esc = re.sub(r"\.\s+", '.<break time="220ms"/> ', t_esc)
    # Slightly slower rate for more natural delivery (tweak to taste)
    return f"<speak><prosody rate='0.98'>{t_esc}</prosody></speak>"

async def punctuate_with_llm(text: str) -> str:
    # Run a lightweight punctuation pass to improve prosody (final-only)
    if not PUNCTUATE_WITH_LLM:
        return text
    try:
        resp = await openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "Punctuate this text for spoken TTS output. Do not add new facts. Keep it concise."},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            max_tokens=max(3, int(len(text) * 0.6))
        )
        punct = resp.choices[0].message["content"].strip()
        # Minimal cleanup: collapse weird punctuation sequences
        punct = re.sub(r"\s+([,\.!?])", r"\1", punct)
        punct = re.sub(r"([,\.!?]){2,}", r"\1", punct)
        return punct
    except Exception as e:
        log.debug(f"Punctuation LLM failed: {e}")
        return text

# =====================================================
# WEBSOCKET HANDLER (based on your v5): single reader, cancellable TTS
# - If NATURAL_SPEECH=1: wait for full assistant response, run punctuation (if enabled),
#   then emit a single TTS audio (more natural).
# - If NATURAL_SPEECH=0: keep prior chunked-TTS streaming behavior for lower latency.
# =====================================================
@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):
    await ws.accept()

    user_id = "solomon_roth"
    recent_msgs = []
    processed_messages = set()

    chat_history = []
    turn_id = 0
    current_active_turn_id = 0

    calendar_kw = ["calendar", "meeting", "schedule", "appointment"]
    plate_kw = ["plate", "add", "to-do", "task", "notion", "list"]

    prompt = await get_notion_prompt()
    greet = prompt.splitlines()[0] if prompt else "Hello Solomon, I’m Silas."

    # GREETING
    try:
        log.info("👋 Sending greeting TTS")
        tts_input = make_ssml_from_text(greet) if USE_SSML else greet
        tts_greet = await openai_client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=tts_input
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
                        log.info(f"🧠 Deepgram partial/final transcript: {transcript}")
                        await dg_queue.put(transcript)

                except Exception as e:
                    log.error(f"❌ DG parse error: {e}")
        except Exception as e:
            log.error(f"❌ DG listener fatal: {e}")

    asyncio.create_task(deepgram_listener_task())

    last_audio_time = time.time()

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

    # Light single-reader for interrupts (client already sends them); we expect this pattern from v5
    async def ws_text_listener():
        nonlocal turn_id, current_active_turn_id
        try:
            while True:
                try:
                    msg = await ws.receive_text()
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    # small sleep and retry
                    await asyncio.sleep(0.05)
                    continue
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                typ = data.get("type")
                if typ == "interrupt":
                    turn_id += 1
                    current_active_turn_id = turn_id
                    log.info(f"⏹️ Received interrupt from client — new active turn {current_active_turn_id}")
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.error(f"ws_text_listener fatal: {e}")

    text_task = asyncio.create_task(ws_text_listener())

    # TRANSCRIPT PROCESSOR
    async def transcript_processor():
        nonlocal last_audio_time, turn_id, current_active_turn_id, chat_history, recent_msgs, processed_messages
        try:
            while True:
                transcript = await dg_queue.get()
                if not transcript:
                    continue

                if len(transcript) < 3 or not any(ch.isalpha() for ch in transcript):
                    continue

                msg = transcript
                norm = _normalize(msg)
                now = time.time()
                recent_msgs = [(m, t) for (m, t) in recent_msgs if now - t < 2]
                if any(_is_similar(m, norm) for (m, t) in recent_msgs):
                    continue
                recent_msgs.append((norm, now))

                # user turn recorded
                chat_history.append({"role": "user", "content": msg})

                # new turn id
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

                # plate / calendar shortcuts (unchanged)
                if any(k in lower for k in plate_kw):
                    if msg in processed_messages:
                        continue
                    processed_messages.add(msg)
                    reply = await send_to_n8n(N8N_PLATE_URL, msg)
                    if current_turn != current_active_turn_id:
                        continue
                    try:
                        tts_payload = make_ssml_from_text(reply) if USE_SSML else reply
                        tts = await openai_client.audio.speech.create(
                            model="gpt-4o-mini-tts",
                            voice="alloy",
                            input=tts_payload
                        )
                        await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                        await ws.send_bytes(await tts.aread())
                    except Exception as e:
                        log.error(f"❌ Plate TTS error: {e}")
                    continue

                if any(k in lower for k in calendar_kw):
                    reply = await send_to_n8n(N8N_CALENDAR_URL, msg)
                    if current_turn != current_active_turn_id:
                        continue
                    try:
                        tts_payload = make_ssml_from_text(reply) if USE_SSML else reply
                        tts = await openai_client.audio.speech.create(
                            model="gpt-4o-mini-tts",
                            voice="alloy",
                            input=tts_payload
                        )
                        await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                        await ws.send_bytes(await tts.aread())
                    except Exception as e:
                        log.error(f"❌ Calendar TTS error: {e}")
                    continue

                # GENERAL GPT streaming
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

                        # If NATURAL_SPEECH is disabled, keep old behavior: generate TTS for chunks
                        if not NATURAL_SPEECH:
                            if len(buffer) >= CHUNK_CHAR_THRESHOLD:
                                if current_turn != current_active_turn_id:
                                    break
                                try:
                                    log.info(f"🎙️ TTS CHUNK START turn={current_turn}, len={len(buffer)}")
                                    tts_payload = make_ssml_from_text(buffer) if USE_SSML else buffer
                                    tts = await openai_client.audio.speech.create(
                                        model="gpt-4o-mini-tts",
                                        voice="alloy",
                                        input=tts_payload
                                    )
                                    if current_turn != current_active_turn_id:
                                        break
                                    await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                                    await ws.send_bytes(await tts.aread())
                                    log.info(f"🎙️ TTS CHUNK SENT turn={current_turn}")
                                except Exception as e:
                                    log.error(f"❌ TTS stream-chunk error: {e}")
                                buffer = ""
                        else:
                            # NATURAL_SPEECH=1: we intentionally do NOT emit TTS for intermediate chunks.
                            # This keeps intermediate audio from breaking sentence flow.
                            # Continue collecting assistant_full_text until stream completes.
                            pass

                    # After streaming ends: handle final output
                    if current_turn == current_active_turn_id:
                        final_text = assistant_full_text.strip()
                        if final_text:
                            # Option A: NATURAL_SPEECH=1 -> do one TTS for the whole final_text (best naturalness)
                            if NATURAL_SPEECH:
                                try:
                                    # optional punctuation pass
                                    if PUNCTUATE_WITH_LLM:
                                        try:
                                            final_text = await punctuate_with_llm(final_text)
                                        except Exception as e:
                                            log.debug(f"Punctuate final failed: {e}")

                                    tts_payload = make_ssml_from_text(final_text) if USE_SSML else final_text
                                    log.info(f"🎙️ TTS FINAL START turn={current_turn}, len={len(final_text)}")
                                    tts = await openai_client.audio.speech.create(
                                        model="gpt-4o-mini-tts",
                                        voice="alloy",
                                        input=tts_payload
                                    )
                                    # check again after generation
                                    if current_turn == current_active_turn_id:
                                        await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                                        await ws.send_bytes(await tts.aread())
                                        log.info(f"🎙️ TTS FINAL SENT turn={current_turn}")
                                except Exception as e:
                                    log.error(f"❌ TTS final-chunk error: {e}")
                            else:
                                # NATURAL_SPEECH=0: we may have leftover buffer from stream; emit it
                                if buffer.strip():
                                    try:
                                        tts_payload = make_ssml_from_text(buffer) if USE_SSML else buffer
                                        tts = await openai_client.audio.speech.create(
                                            model="gpt-4o-mini-tts",
                                            voice="alloy",
                                            input=tts_payload
                                        )
                                        if current_turn == current_active_turn_id:
                                            await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                                            await ws.send_bytes(await tts.aread())
                                    except Exception as e:
                                        log.error(f"❌ TTS final-chunk error (non-natural): {e}")

                            # store assistant in history (for context)
                            chat_history.append({"role": "assistant", "content": final_text})
                            log.info(f"💾 Stored assistant turn {current_turn} in history (len={len(chat_history)})")

                    asyncio.create_task(mem0_add(user_id, msg))

                except Exception as e:
                    log.error(f"LLM error: {e}")

        except Exception as e:
            log.error(f"❌ transcript_processor fatal: {e}")

    transcript_task = asyncio.create_task(transcript_processor())

    # MAIN LOOP — browser audio -> Deepgram (unchanged)
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
                audio_bytes = audio_bytes + b"\x00"

            last_audio_time = time.time()

            try:
                import struct as _struct
                if len(audio_bytes) >= 20:
                    samples = _struct.unpack("<10h", audio_bytes[:20])
                    log.info(f"PCM samples[0:10] = {list(samples)}")
            except Exception as e:
                log.error(f"sample unpack error: {e}")

            log.info(f"📡 PCM audio received — {len(audio_bytes)} bytes")

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
            text_task.cancel()
        except:
            pass
        try:
            await dg_ws.close()
        except:
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
