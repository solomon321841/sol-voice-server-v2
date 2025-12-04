import os
import json
import logging
import asyncio
import time
from typing import List, Dict
from dotenv import load_dotenv
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from openai import AsyncOpenAI
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
# ENDPOINTS / MODEL
# =====================================================
N8N_CALENDAR_URL = "https://n8n.marshall321.org/webhook/calendar-agent"
N8N_PLATE_URL = "https://n8n.marshall321.org/webhook/agent/plate"

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
# Helpers (mem0, notion, n8n) - unchanged/simple
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
# Normalization helpers
# =====================================================
def _normalize(m: str):
    m = m.lower().strip()
    m = "".join(ch for ch in m if ch not in string.punctuation)
    return " ".join(m.split())

def _is_similar(a: str, b: str):
    return bool(a and b and (a == b or a.startswith(b) or b.startswith(a) or a in b or b in a))

# =====================================================
# Websocket handler (streaming + interrupt + debounce of transcripts)
# =====================================================
@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):
    await ws.accept()

    user_id = "solomon_roth"
    recent_msgs = []
    processed_messages = set()
    calendar_kw = ["calendar", "meeting", "schedule", "appointment"]
    plate_kw = ["plate", "add", "to-do", "task", "notion", "list"]

    prompt = await get_notion_prompt()
    greet = prompt.splitlines()[0] if prompt else "Hello Solomon, I’m Silas."

    # Conversation context
    session_messages: List[Dict] = [{"role": "system", "content": prompt}]

    # greeting
    try:
        tts_greet = await openai_client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=greet
        )
        await ws.send_bytes(await tts_greet.aread())
    except Exception as e:
        log.error(f"Greeting TTS error: {e}")

    # =====================================================
    # connect to Deepgram
    # =====================================================
    if not DEEPGRAM_API_KEY:
        log.error("No DEEPGRAM_API_KEY set.")
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
        log.error(f"Failed to connect to Deepgram WS: {e}")
        return

    # =====================================================
    # Deepgram listener: put (transcript, is_final) into dg_queue
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

                    transcript = ""
                    is_final = False

                    if "channel" in data and isinstance(data["channel"], dict):
                        alts = data["channel"].get("alternatives", [])
                        if alts and isinstance(alts[0], dict):
                            transcript = alts[0].get("transcript", "").strip()
                            is_final = bool(alts[0].get("is_final") or data.get("type") in ("UtteranceEnd", "UtteranceFinal"))
                    elif "results" in data and isinstance(data["results"], dict):
                        ch = data["results"].get("channels", [])
                        if ch and isinstance(ch, list):
                            alts = ch[0].get("alternatives", [])
                        else:
                            alts = data["results"].get("alternatives", [])
                        if alts and isinstance(alts[0], dict):
                            transcript = alts[0].get("transcript", "").strip()
                            is_final = bool(alts[0].get("is_final") or data.get("is_final"))
                    else:
                        if isinstance(data.get("transcript"), str):
                            transcript = data.get("transcript").strip()
                            is_final = bool(data.get("is_final"))

                    if transcript:
                        await dg_queue.put((transcript, bool(is_final)))
                except Exception as e:
                    log.error(f"DG parse error: {e}")
        except Exception as e:
            log.error(f"DG listener fatal: {e}")

    dg_listener = asyncio.create_task(deepgram_listener_task())

    # keepalive to Deepgram
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
                        log.error(f"Error sending keepalive to Deepgram: {e}")
                        break
        except asyncio.CancelledError:
            return
    keepalive_task = asyncio.create_task(dg_keepalive_task())

    # =====================================================
    # cancellable tasks
    # =====================================================
    current_stream_task = None
    current_tts_task = None

    async def send_tts_bytes(tts_audio_creator):
        nonlocal current_tts_task
        try:
            audio_bytes = await tts_audio_creator()
            if not audio_bytes:
                return
            try:
                await ws.send_bytes(audio_bytes)
                log.info("TTS chunk sent to client (len=%d)", len(audio_bytes))
            except Exception as e:
                log.error(f"Error sending TTS bytes to client: {e}")
        except asyncio.CancelledError:
            log.info("send_tts_bytes: cancelled (client interrupt)")
        except Exception as e:
            log.error(f"send_tts_bytes error: {e}")
        finally:
            if asyncio.current_task() is current_tts_task:
                current_tts_task = None

    # =====================================================
    # LLM streaming + incremental TTS producer
    # =====================================================
    async def run_llm_stream_and_tts(user_message_text: str):
        nonlocal current_tts_task, current_stream_task, session_messages

        session_messages.append({"role": "user", "content": user_message_text})

        try:
            stream = await openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=session_messages,
                stream=True,
            )
        except Exception as e:
            log.error(f"Error starting LLM stream: {e}")
            return

        full_response = ""
        last_sent_idx = 0

        MIN_CHARS_FOR_TTS = 60
        SEND_ON_PUNCTUATION = True

        async def maybe_send_chunk():
            nonlocal last_sent_idx, full_response, current_tts_task
            if last_sent_idx >= len(full_response):
                return
            candidate = full_response[last_sent_idx:].strip()
            if not candidate:
                return
            if len(candidate) < MIN_CHARS_FOR_TTS:
                if not SEND_ON_PUNCTUATION or candidate[-1] not in ".!?":
                    return
            async def make_tts_for(text=candidate):
                try:
                    tts = await openai_client.audio.speech.create(
                        model="gpt-4o-mini-tts",
                        voice="alloy",
                        input=text
                    )
                    return await tts.aread()
                except Exception as e:
                    log.error(f"Error generating TTS: {e}")
                    return b""
            if current_tts_task and not current_tts_task.done():
                try:
                    current_tts_task.cancel()
                    log.info("Cancelled previous TTS task before sending new chunk")
                except Exception as e:
                    log.error(f"Error cancelling current_tts_task: {e}")
            current_tts_task = asyncio.create_task(send_tts_bytes(make_tts_for))
            last_sent_idx = len(full_response)

        try:
            async for chunk in stream:
                delta = getattr(chunk.choices[0].delta, "content", "")
                if delta:
                    full_response += delta
                    await maybe_send_chunk()
                if asyncio.current_task().cancelled():
                    log.info("LLM stream task cancelled mid-stream")
                    break

            # final remainder
            if len(full_response.strip()) > last_sent_idx:
                remainder = full_response[last_sent_idx:].strip()
                if remainder:
                    if current_tts_task and not current_tts_task.done():
                        try:
                            current_tts_task.cancel()
                        except Exception as e:
                            log.error(f"Error cancelling current_tts_task: {e}")
                    async def make_tts_final(text=remainder):
                        try:
                            tts = await openai_client.audio.speech.create(
                                model="gpt-4o-mini-tts",
                                voice="alloy",
                                input=text
                            )
                            return await tts.aread()
                        except Exception as e:
                            log.error(f"Error generating final TTS: {e}")
                            return b""
                    current_tts_task = asyncio.create_task(send_tts_bytes(make_tts_final))
                    last_sent_idx = len(full_response)

            assistant_content = full_response.strip()
            if assistant_content:
                session_messages.append({"role": "assistant", "content": assistant_content})
            return
        except asyncio.CancelledError:
            log.info("run_llm_stream_and_tts: cancelled due to client interrupt")
            if current_tts_task and not current_tts_task.done():
                try:
                    current_tts_task.cancel()
                except Exception as e:
                    log.error(f"Error cancelling current_tts_task on stream cancel: {e}")
            return
        except Exception as e:
            log.error(f"LLM stream error: {e}")
            return

    # =====================================================
    # Transcript processing with DEBOUNCE logic to avoid reacting to every partial
    # =====================================================
    DEBOUNCE_MS = 400         # milliseconds to wait for more partial transcripts before processing
    MIN_WORDS_TO_PROCESS = 1  # minimum words to accept for processing (adjust as needed)

    pending_transcript = None
    pending_is_final = False
    debounce_task = None
    debounce_lock = asyncio.Lock()

    async def process_pending_transcript():
        nonlocal pending_transcript, pending_is_final, current_stream_task, current_tts_task
        async with debounce_lock:
            if not pending_transcript:
                return
            text = pending_transcript.strip()
            pending_transcript = None
            pending_is_final = False

        # basic sanity
        if not text or len(text) < 1:
            return
        if sum(1 for w in text.split() if w.strip()) < MIN_WORDS_TO_PROCESS:
            log.info("Skipping very short transcript")
            return

        log.info(f"Processing transcript (debounced): {text}")

        # cancel any running stream / tts to prioritize this new utterance
        if current_stream_task and not current_stream_task.done():
            try:
                current_stream_task.cancel()
                log.info("Cancelled ongoing LLM stream due to new transcript")
            except Exception as e:
                log.error(f"Error cancelling current_stream_task: {e}")
        if current_tts_task and not current_tts_task.done():
            try:
                current_tts_task.cancel()
                log.info("Cancelled outgoing TTS due to new transcript")
            except Exception as e:
                log.error(f"Error cancelling current_tts_task: {e}")

        # start a new LLM streaming task (background)
        current_stream_task = asyncio.create_task(run_llm_stream_and_tts(text))

    async def schedule_debounce_and_process():
        nonlocal debounce_task
        # cancel previous debounce
        if debounce_task and not debounce_task.done():
            debounce_task.cancel()
        async def _wait_and_run():
            try:
                await asyncio.sleep(DEBOUNCE_MS / 1000.0)
                await process_pending_transcript()
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error(f"debounce inner error: {e}")
        debounce_task = asyncio.create_task(_wait_and_run())

    async def transcript_processor():
        nonlocal pending_transcript, pending_is_final
        try:
            while True:
                try:
                    transcript, is_final = await dg_queue.get()
                except asyncio.CancelledError:
                    break

                if not transcript:
                    continue

                log.info(f"DG transcript (final={is_final}): {transcript}")

                # quick filter
                if len(transcript.strip()) < 1 or not any(ch.isalpha() for ch in transcript):
                    continue

                # normalization + dedupe
                norm = _normalize(transcript)
                now = time.time()
                nonlocal_recent = [(m, t) for (m, t) in recent_msgs if now - t < 2]
                # use existing recent_msgs array
                recent_msgs[:] = nonlocal_recent
                if any(_is_similar(m, norm) for (m, t) in recent_msgs):
                    log.info("Duplicate/very-similar transcript — ignoring")
                    continue
                recent_msgs.append((norm, now))

                # If final, process immediately
                if is_final:
                    # set pending and cancel debounce to process right away
                    async with debounce_lock:
                        pending_transcript = transcript
                        pending_is_final = True
                    if debounce_task and not debounce_task.done():
                        debounce_task.cancel()
                    await process_pending_transcript()
                    continue

                # Not final -> update pending_transcript and schedule debounce
                async with debounce_lock:
                    # append or replace with latest partial (we keep latest partial)
                    pending_transcript = transcript
                    pending_is_final = False
                await schedule_debounce_and_process()

        except Exception as e:
            log.error(f"transcript_processor fatal: {e}")

    transcript_task = asyncio.create_task(transcript_processor())

    # =====================================================
    # Single ws reader & main loop: same robust pattern used earlier
    # =====================================================
    incoming_queue = asyncio.Queue()

    async def ws_reader_task():
        try:
            while True:
                try:
                    data = await ws.receive()
                except WebSocketDisconnect:
                    log.info("ws_reader_task: WebSocketDisconnect")
                    await incoming_queue.put({"type": "disconnect", "reason": "WebSocketDisconnect"})
                    break
                except RuntimeError as e:
                    msg = str(e)
                    log.info(f"ws_reader_task RuntimeError: {msg}")
                    await incoming_queue.put({"type": "disconnect", "reason": msg})
                    break
                except Exception as e:
                    msg = str(e)
                    log.error(f"ws_reader_task receive error: {msg}")
                    await incoming_queue.put({"type": "disconnect", "reason": msg})
                    break
                await incoming_queue.put(data)
        except asyncio.CancelledError:
            log.info("ws_reader_task cancelled")
        finally:
            log.info("ws_reader_task exiting")

    reader = asyncio.create_task(ws_reader_task())

    # =====================================================
    # MAIN: handle incoming_queue: control frames + binary audio
    # control example: {"control":"interrupt"}
    # =====================================================
    try:
        while True:
            item = await incoming_queue.get()
            if not item:
                continue
            if isinstance(item, dict) and item.get("type") == "disconnect":
                log.info(f"Reader reported disconnect: {item.get('reason')}")
                break
            data = item

            # handle text control messages
            if data.get("type") == "websocket.receive" and data.get("text") is not None:
                txt = data["text"]
                try:
                    obj = json.loads(txt)
                    if isinstance(obj, dict) and obj.get("control") == "interrupt":
                        log.info("Client interrupt control received — cancelling current stream & TTS.")
                        if current_stream_task and not current_stream_task.done():
                            try:
                                current_stream_task.cancel()
                                log.info("Requested cancellation of current_stream_task")
                            except Exception as e:
                                log.error(f"Error cancelling current_stream_task: {e}")
                        if current_tts_task and not current_tts_task.done():
                            try:
                                current_tts_task.cancel()
                                log.info("Requested cancellation of current_tts_task")
                            except Exception as e:
                                log.error(f"Error cancelling current_tts_task: {e}")
                        # clear pending transcript so we don't resume processing an old partial
                        async with debounce_lock:
                            nonlocal pending_transcript  # type: ignore
                            pending_transcript = None
                        # also cancel debounce
                        if debounce_task and not debounce_task.done():
                            debounce_task.cancel()
                        continue
                except Exception:
                    pass

            # binary audio frames
            if data.get("type") != "websocket.receive" or data.get("bytes") is None:
                continue

            audio_bytes = data["bytes"]
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes + b'\x00'

            last_audio_time = time.time()

            # PCM logs / stats
            import struct
            try:
                if len(audio_bytes) >= 20:
                    samples = struct.unpack("<10h", audio_bytes[:20])
                    log.debug(f"PCM samples[0:10] = {list(samples)}")
            except Exception as e:
                log.debug(f"sample unpack error: {e}")

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

            # forward to Deepgram
            try:
                await dg_ws.send(audio_bytes)
            except Exception as e:
                log.error(f"Error sending audio to Deepgram WS: {e}")
                continue

    except Exception as e:
        log.error(f"Main loop unexpected error: {e}")
    finally:
        log.info("Main websocket handler cleanup")
        try:
            reader.cancel()
        except:
            pass
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
            if debounce_task and not debounce_task.done():
                debounce_task.cancel()
        except:
            pass
        try:
            if current_stream_task:
                current_stream_task.cancel()
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
