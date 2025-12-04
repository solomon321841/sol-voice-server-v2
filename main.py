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
# Helpers (mem0, notion, n8n) - unchanged behaviour
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
# Normalization
# =====================================================
def _normalize(m: str):
    m = m.lower().strip()
    m = "".join(ch for ch in m if ch not in string.punctuation)
    return " ".join(m.split())

def _is_similar(a: str, b: str):
    return bool(a and b and (a == b or a.startswith(b) or b.startswith(a) or a in b or b in a))

# =====================================================
# MAIN WEBSOCKET HANDLER
# - Supports:
#   * streaming LLM -> incremental TTS (fast)
#   * cancellable LLM stream and cancellable TTS sends on client interrupt
#   * robust ws reader and deepgram forwarding
# =====================================================
@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):
    await ws.accept()

    # session variables
    user_id = "solomon_roth"
    recent_msgs = []
    processed_messages = set()
    calendar_kw = ["calendar", "meeting", "schedule", "appointment"]
    plate_kw = ["plate", "add", "to-do", "task", "notion", "list"]

    prompt = await get_notion_prompt()
    greet = prompt.splitlines()[0] if prompt else "Hello Solomon, I’m Silas."

    # conversation history for context (system prompt as first message)
    session_messages: List[Dict] = [{"role": "system", "content": prompt}]

    # greeting TTS
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
    # Deepgram connect
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
    # Deepgram listener: push transcripts into dg_queue
    # - we'll accept both partial and final, but tag them so transcript_processor can decide
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

                    # Extract transcript if present
                    transcript = ""
                    is_final = False

                    # Try standard shapes
                    if "channel" in data and isinstance(data["channel"], dict):
                        alts = data["channel"].get("alternatives", [])
                        if alts and isinstance(alts[0], dict):
                            transcript = alts[0].get("transcript", "").strip()
                            is_final = bool(alts[0].get("is_final") or data.get("type") in ("UtteranceEnd", "UtteranceFinal"))
                    elif "results" in data and isinstance(data["results"], dict):
                        alts = data["results"].get("alternatives", [])
                        if alts and isinstance(alts[0], dict):
                            transcript = alts[0].get("transcript", "").strip()
                            is_final = bool(alts[0].get("is_final") or data.get("is_final"))
                    else:
                        if isinstance(data.get("transcript"), str):
                            transcript = data.get("transcript").strip()
                            is_final = bool(data.get("is_final"))

                    if transcript:
                        # push tuple (transcript, is_final)
                        await dg_queue.put((transcript, bool(is_final)))
                except Exception as e:
                    log.error(f"❌ DG parse error: {e}")
        except Exception as e:
            log.error(f"❌ DG listener fatal: {e}")

    dg_listener = asyncio.create_task(deepgram_listener_task())

    # =====================================================
    # Backend keepalive to Deepgram
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
    # Cancellable tasks (for interruption behavior)
    # - current_stream_task: running LLM streaming -> produces incremental text
    # - current_tts_task: sending generated TTS bytes to client (cancellable)
    # =====================================================
    current_stream_task = None
    current_tts_task = None

    async def send_tts_bytes(tts_audio_creator):
        """
        Await tts_audio_creator() to get bytes, then send them to the client.
        This is cancellable via the task that wraps this function.
        """
        nonlocal current_tts_task
        try:
            audio_bytes = await tts_audio_creator()
            if not audio_bytes:
                return
            try:
                await ws.send_bytes(audio_bytes)
                log.info("TTS chunk sent to client (len=%d)", len(audio_bytes))
            except Exception as e:
                log.error(f"❌ Error sending TTS bytes to client: {e}")
        except asyncio.CancelledError:
            log.info("send_tts_bytes: cancelled (client interrupted)")
        except Exception as e:
            log.error(f"❌ send_tts_bytes error: {e}")
        finally:
            if asyncio.current_task() is current_tts_task:
                current_tts_task = None

    # =====================================================
    # Helper: process a single user message by streaming LLM and producing incremental TTS
    # - Maintains session_messages for context
    # - Sends TTS incrementally: takes text from last_sent_index -> current buffer when
    #   text length exceeds thresholds or ends with punctuation.
    # - Honors cancellation: cancelling current_stream_task will stop generating more TTS.
    # =====================================================
    async def run_llm_stream_and_tts(user_message_text: str):
        nonlocal current_tts_task, current_stream_task, session_messages

        # Add user's utterance into session history (so LLM has context)
        session_messages.append({"role": "user", "content": user_message_text})

        # We'll stream the LLM response and send incremental TTS parts.
        try:
            # Start streaming completion
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

        # heuristics for when to send a TTS chunk
        MIN_CHARS_FOR_TTS = 60       # minimum chars to wait before generating TTS chunk
        SEND_ON_PUNCTUATION = True   # send if chunk ends with .,!? to reduce mid-sentence chunks

        async def maybe_send_chunk():
            nonlocal last_sent_idx, full_response, current_tts_task
            # compute candidate substring
            if last_sent_idx >= len(full_response):
                return
            candidate = full_response[last_sent_idx:].strip()
            if not candidate:
                return

            # Decide whether to send now
            if len(candidate) < MIN_CHARS_FOR_TTS:
                # try to send if ends with punctuation and allowed
                if not SEND_ON_PUNCTUATION or candidate[-1] not in ".!?":
                    return

            # Prepare TTS bytes for this candidate substring
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

            # Cancel previous outgoing TTS (we stream and show latest)
            if current_tts_task and not current_tts_task.done():
                try:
                    current_tts_task.cancel()
                    log.info("Cancelled previous TTS task before sending new chunk")
                except Exception as e:
                    log.error(f"Error cancelling current_tts_task: {e}")

            # Spawn task to send bytes (cancellable)
            current_tts_task = asyncio.create_task(send_tts_bytes(make_tts_for))
            # advance last_sent_idx to end of full_response (we consider we've sent it)
            last_sent_idx = len(full_response)

        try:
            # iterate stream chunks
            async for chunk in stream:
                # extract delta content
                delta = getattr(chunk.choices[0].delta, "content", "")
                if delta:
                    full_response += delta
                    # consider sending incremental TTS if thresholds met
                    await maybe_send_chunk()
                # If task cancellation requested, break early
                if asyncio.current_task().cancelled():
                    log.info("LLM stream task cancelled mid-stream")
                    break

            # After stream completes, send any remaining text
            if len(full_response.strip()) > last_sent_idx:
                # final remainder
                remainder = full_response[last_sent_idx:].strip()
                if remainder:
                    # cancel previous TTS (to ensure no overlap)
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
                    # mark that we've sent all
                    last_sent_idx = len(full_response)

            # Append assistant reply to session history for context
            assistant_content = full_response.strip()
            if assistant_content:
                session_messages.append({"role": "assistant", "content": assistant_content})
                # Optionally store to mem0
            return
        except asyncio.CancelledError:
            # If we are cancelled because client interrupted, stop generating and return.
            log.info("run_llm_stream_and_tts: cancelled due to client interrupt")
            # Cancel any outstanding TTS send
            if current_tts_task and not current_tts_task.done():
                try:
                    current_tts_task.cancel()
                except Exception as e:
                    log.error(f"Error cancelling current_tts_task on stream cancel: {e}")
            # Do not append partial assistant text to session history (avoids sticking partially)
            return
        except Exception as e:
            log.error(f"LLM stream error: {e}")
            return

    # =====================================================
    # Transcript processor: consumes dg_queue items (transcript, is_final)
    # - For speed we will start processing on the first useful transcript (even partial).
    # - If the client sends an explicit interrupt, we cancel current_stream_task & current_tts_task.
    # =====================================================
    async def transcript_processor():
        nonlocal current_stream_task, current_tts_task, session_messages, recent_msgs
        try:
            while True:
                try:
                    transcript, is_final = await dg_queue.get()
                except asyncio.CancelledError:
                    break

                if not transcript:
                    continue

                # rate-limit / dedupe
                log.info(f"📝 DG transcript (final={is_final}): {transcript}")
                if not transcript or len(transcript) < 3 or not any(ch.isalpha() for ch in transcript):
                    continue

                norm = _normalize(transcript)
                now = time.time()
                recent_msgs = [(m, t) for (m, t) in recent_msgs if now - t < 2]
                if any(_is_similar(m, norm) for (m, t) in recent_msgs):
                    continue
                recent_msgs.append((norm, now))

                # Immediately start processing the user transcript to be fast (stream LLM)
                # Cancel any existing LLM stream or outgoing TTS (we treat this as a new user utterance)
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

                # Kick off LLM streaming + incremental TTS for this utterance
                # We don't await here; run in background so processor can keep consuming Deepgram partials
                current_stream_task = asyncio.create_task(run_llm_stream_and_tts(transcript))

        except Exception as e:
            log.error(f"❌ transcript_processor fatal: {e}")

    transcript_task = asyncio.create_task(transcript_processor())

    # =====================================================
    # SINGLE WS READER: isolate ws.receive into one task and feed incoming_queue
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
                await incoming_queue.put(data)
        except asyncio.CancelledError:
            log.info("ws_reader_task cancelled")
        finally:
            log.info("ws_reader_task exiting")

    reader = asyncio.create_task(ws_reader_task())

    # =====================================================
    # MAIN loop: consume incoming_queue (control frames, audio bytes)
    # - Accept control frames: {"control":"interrupt"} to cancel current_stream_task & current_tts_task
    # - Forward binary audio to Deepgram
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
            # Handle text control frames
            if data.get("type") == "websocket.receive" and data.get("text") is not None:
                txt = data["text"]
                try:
                    obj = json.loads(txt)
                    if isinstance(obj, dict) and obj.get("control") == "interrupt":
                        log.info("🔴 Client interrupt control received — cancelling current stream & TTS.")
                        # cancel LLM streaming
                        if current_stream_task and not current_stream_task.done():
                            try:
                                current_stream_task.cancel()
                                log.info("Requested cancellation of current_stream_task")
                            except Exception as e:
                                log.error(f"Error cancelling current_stream_task: {e}")
                        # cancel outgoing TTS
                        if current_tts_task and not current_tts_task.done():
                            try:
                                current_tts_task.cancel()
                                log.info("Requested cancellation of current_tts_task")
                            except Exception as e:
                                log.error(f"Error cancelling current_tts_task: {e}")
                        # Clear any server-side queued TTS (we don't maintain a separate queue)
                        continue
                except Exception:
                    # ignore non-control text
                    pass

            # binary audio frames
            if data.get("type") != "websocket.receive" or data.get("bytes") is None:
                continue

            audio_bytes = data["bytes"]

            # ensure even-length (16-bit alignment)
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes + b'\x00'

            last_audio_time = time.time()

            # PCM logging
            import struct
            try:
                if len(audio_bytes) >= 20:
                    samples = struct.unpack("<10h", audio_bytes[:20])
                    log.debug(f"PCM samples[0:10] = {list(samples)}")
            except Exception as e:
                log.debug(f"sample unpack error: {e}")

            log.info(f"📡 PCM audio received — {len(audio_bytes)} bytes")

            # PCM stats
            try:
                if len(audio_bytes) >= 2:
                    total_samples = len(audio_bytes) // 2
                    all_samples = struct.unpack("<" + "h" * total_samples, audio_bytes[: total_samples * 2])
                    peak = max(abs(s) for s in all_samples) if all_samples else 0
                    rms = (sum(s * s for s in all_samples) / total_samples) ** 0.5 if total_samples > 0 else 0
                    log.info(f"🔊 PCM STATS — RMS={rms:.2f}, Peak={peak}")
            except Exception as e:
                log.error(f"PCM stats error: {e}")

            # Forward raw PCM to Deepgram
            try:
                await dg_ws.send(audio_bytes)
            except Exception as e:
                log.error(f"❌ Error sending audio to Deepgram WS: {e}")
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
