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
from asyncio import Queue

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
# HELPERS (unchanged from your prior file)
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
# NORMALIZATION
# =====================================================
def _normalize(m: str):
    m = m.lower().strip()
    m = "".join(ch for ch in m if ch not in string.punctuation)
    return " ".join(m.split())

def _is_similar(a: str, b: str):
    return bool(a and b and (a == b or a.startswith(b) or b.startswith(a) or a in b or b in a))

# =====================================================
# WEBSOCKET HANDLER (robust receive via a single reader task)
# =====================================================
@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):
    await ws.accept()

    user_id = "solomon_roth"
    recent_msgs = []
    processed_messages = set()

    calendar_kw = ["calendar", "meeting", "schedule", "appointment"]
    plate_kw = ["plate", "add", "to-do", "task", "notion", "list"]

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

    # GREETING TTS
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
    # DEEPGRAM LISTENER -> pushes transcripts into dg_queue
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
                    else:
                        pass

                    transcript = ""
                    if alts and isinstance(alts, list):
                        transcript = alts[0].get("transcript", "").strip()

                    if transcript:
                        await dg_queue.put(transcript)

                except Exception as e:
                    log.error(f"❌ DG parse error: {e}")
        except Exception as e:
            log.error(f"❌ DG listener fatal: {e}")

    dg_listener = asyncio.create_task(deepgram_listener_task())

    # =====================================================
    # BACKEND KEEPALIVE
    # =====================================================
    last_audio_time = time.time()

    async def dg_keepalive_task():
        nonlocal last_audio_time
        try:
            while True:
                await asyncio.sleep(1.2)
                if time.time() - last_audio_time > 1.5:
                    try:
                        silence = (b'\x00\x00') * 4800
                        await dg_ws.send(silence)
                    except Exception as e:
                        log.error(f"❌ Error sending keepalive to Deepgram: {e}")
                        break
        except asyncio.CancelledError:
            return

    keepalive_task = asyncio.create_task(dg_keepalive_task())

    # =====================================================
    # CANCELLABLE TTS SENDING SUPPORT
    # =====================================================
    current_tts_task = None

    async def send_tts_bytes(tts_audio_creator):
        nonlocal current_tts_task
        try:
            log.info("TTS send task started")
            audio_bytes = await tts_audio_creator()
            if not audio_bytes:
                log.info("TTS generator returned empty bytes")
                return
            try:
                await ws.send_bytes(audio_bytes)
                log.info("TTS bytes successfully sent to client")
            except Exception as e:
                log.error(f"❌ Error sending TTS bytes to client: {e}")
        except asyncio.CancelledError:
            log.info("TTS send task cancelled due to client interrupt.")
        except Exception as e:
            log.error(f"❌ send_tts_bytes error: {e}")
        finally:
            if asyncio.current_task() is current_tts_task:
                current_tts_task = None

    # =====================================================
    # TRANSCRIPT PROCESSOR (consumes dg_queue), uses send_tts_bytes for TTS
    # =====================================================
    async def transcript_processor():
        nonlocal recent_msgs, processed_messages, prompt, current_tts_task
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

                mems = await mem0_search(user_id, msg)
                ctx = memory_context(mems)
                sys_prompt = f"{prompt}\n\nFacts:\n{ctx}"
                lower = msg.lower()

                # PLATE LOGIC
                if any(k in lower for k in plate_kw):
                    if msg in processed_messages:
                        continue
                    processed_messages.add(msg)

                    reply = await send_to_n8n(N8N_PLATE_URL, msg)

                    try:
                        async def make_tts():
                            tts = await openai_client.audio.speech.create(
                                model="gpt-4o-mini-tts",
                                voice="alloy",
                                input=reply
                            )
                            return await tts.aread()

                        if current_tts_task and not current_tts_task.done():
                            log.info("Cancelling previous TTS before starting plate TTS")
                            try:
                                current_tts_task.cancel()
                            except Exception as e:
                                log.error(f"Error cancelling current TTS task: {e}")

                        current_tts_task = asyncio.create_task(send_tts_bytes(make_tts))
                    except Exception as e:
                        log.error(f"❌ TTS plate error: {e}")
                    continue

                # CALENDAR LOGIC
                if any(k in lower for k in calendar_kw):
                    reply = await send_to_n8n(N8N_CALENDAR_URL, msg)
                    try:
                        async def make_tts():
                            tts = await openai_client.audio.speech.create(
                                model="gpt-4o-mini-tts",
                                voice="alloy",
                                input=reply
                            )
                            return await tts.aread()

                        if current_tts_task and not current_tts_task.done():
                            log.info("Cancelling previous TTS before starting calendar TTS")
                            try:
                                current_tts_task.cancel()
                            except Exception as e:
                                log.error(f"Error cancelling current TTS task: {e}")

                        current_tts_task = asyncio.create_task(send_tts_bytes(make_tts))
                    except Exception as e:
                        log.error(f"❌ TTS calendar error: {e}")
                    continue

                # GENERAL GPT LOGIC
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
                                partial = buffer
                                buffer = ""

                                async def make_tts_partial(p=partial):
                                    tts = await openai_client.audio.speech.create(
                                        model="gpt-4o-mini-tts",
                                        voice="alloy",
                                        input=p
                                    )
                                    return await tts.aread()

                                if current_tts_task and not current_tts_task.done():
                                    log.info("Cancelling previous TTS before starting partial TTS")
                                    try:
                                        current_tts_task.cancel()
                                    except Exception as e:
                                        log.error(f"Error cancelling current TTS task: {e}")

                                current_tts_task = asyncio.create_task(send_tts_bytes(make_tts_partial))

                    if buffer.strip():
                        final_text = buffer

                        async def make_tts_final(p=final_text):
                            tts = await openai_client.audio.speech.create(
                                model="gpt-4o-mini-tts",
                                voice="alloy",
                                input=p
                            )
                            return await tts.aread()

                        if current_tts_task and not current_tts_task.done():
                            log.info("Cancelling previous TTS before starting final TTS")
                            try:
                                current_tts_task.cancel()
                            except Exception as e:
                                log.error(f"Error cancelling current TTS task: {e}")

                        current_tts_task = asyncio.create_task(send_tts_bytes(make_tts_final))

                    asyncio.create_task(mem0_add(user_id, msg))

                except Exception as e:
                    log.error(f"LLM error: {e}")

        except Exception as e:
            log.error(f"❌ transcript_processor fatal: {e}")

    transcript_task = asyncio.create_task(transcript_processor())

    # =====================================================
    # SINGLE READER TASK: reads from ws.receive() and pushes items into incoming_queue
    # This isolates receive() to one location and gracefully handles disconnects.
    # =====================================================
    incoming_queue = asyncio.Queue()

    async def ws_reader_task():
        try:
            while True:
                try:
                    data = await ws.receive()
                except WebSocketDisconnect:
                    log.info("ws_reader_task: WebSocketDisconnect received")
                    await incoming_queue.put({"type": "disconnect", "reason": "WebSocketDisconnect"})
                    break
                except RuntimeError as e:
                    msg = str(e)
                    log.info(f"ws_reader_task RuntimeError during ws.receive(): {msg}")
                    await incoming_queue.put({"type": "disconnect", "reason": msg})
                    break
                except Exception as e:
                    msg = str(e)
                    log.error(f"ws_reader_task unexpected receive error: {msg}")
                    await incoming_queue.put({"type": "disconnect", "reason": msg})
                    break

                # push the received frame into incoming_queue for the main loop to process
                await incoming_queue.put(data)

        except asyncio.CancelledError:
            log.info("ws_reader_task cancelled")
        finally:
            log.info("ws_reader_task exiting")

    reader = asyncio.create_task(ws_reader_task())

    # =====================================================
    # MAIN LOOP: consume incoming_queue rather than calling ws.receive() directly
    # =====================================================
    try:
        while True:
            item = await incoming_queue.get()
            if not item:
                continue

            # handle disconnect sentinel from reader
            if isinstance(item, dict) and item.get("type") == "disconnect":
                log.info(f"Reader reported disconnect: {item.get('reason')}")
                break

            # item is expected to be a dict like Starlette returns from ws.receive()
            data = item

            # handle text control messages
            if data.get("type") == "websocket.receive" and data.get("text") is not None:
                txt = data["text"]
                try:
                    obj = json.loads(txt)
                    if isinstance(obj, dict) and obj.get("control") == "interrupt":
                        log.info("🔴 Client interrupt control received — cancelling current TTS.")
                        if current_tts_task and not current_tts_task.done():
                            try:
                                current_tts_task.cancel()
                                log.info("Requested cancellation of current_tts_task")
                            except Exception as e:
                                log.error(f"Error cancelling current_tts_task: {e}")
                        continue
                except Exception:
                    # Not a control frame - ignore
                    pass

            # binary audio frames handling
            if data.get("type") != "websocket.receive" or data.get("bytes") is None:
                continue

            audio_bytes = data["bytes"]

            # ensure even-length (16-bit alignment)
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes + b'\x00'

            last_audio_time = time.time()

            # PCM SAMPLE LOGGING (small sample)
            import struct
            try:
                if len(audio_bytes) >= 20:
                    samples = struct.unpack("<10h", audio_bytes[:20])
                    log.info(f"PCM samples[0:10] = {list(samples)}")
            except Exception as e:
                log.error(f"sample unpack error: {e}")

            log.info(f"📡 PCM audio received — {len(audio_bytes)} bytes")

            # PCM STATS (RMS/Peak)
            try:
                if len(audio_bytes) >= 2:
                    total_samples = len(audio_bytes) // 2
                    all_samples = struct.unpack("<" + "h" * total_samples, audio_bytes[: total_samples * 2])
                    peak = max(abs(s) for s in all_samples) if all_samples else 0
                    rms = (sum(s * s for s in all_samples) / total_samples) ** 0.5 if total_samples > 0 else 0
                    log.info(f"🔊 PCM STATS — RMS={rms:.2f}, Peak={peak}")
            except Exception as e:
                log.error(f"PCM stats error: {e}")

            # Forward raw PCM bytes to Deepgram
            try:
                await dg_ws.send(audio_bytes)
            except Exception as e:
                log.error(f"❌ Error sending audio to Deepgram WS: {e}")
                continue

    except Exception as e:
        log.error(f"Main loop unexpected error: {e}")
    finally:
        log.info("Main websocket handler cleaning up")
        # cancel reader
        try:
            reader.cancel()
        except Exception:
            pass
        # cancel other tasks
        try:
            keepalive_task.cancel()
        except:
            pass
        try:
            if transcript_task:
                transcript_task.cancel()
        except:
            pass
        try:
            if current_tts_task:
                current_tts_task.cancel()
        except:
            pass
        try:
            dg_listener.cancel()
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
