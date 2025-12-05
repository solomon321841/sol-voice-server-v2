import os
import json
import logging
import asyncio
import time
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
GPT_MODEL = "gpt-5.1"

# TTS chunk size for natural speech
CHUNK_CHAR_THRESHOLD = 90  # tweak between ~60–120

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
# N8N HELPER
# =====================================================
async def send_to_n8n(url: str, message: str) -> str:
    """Send a message to an n8n webhook and return the response."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(url, json={"message": message})
            response.raise_for_status()
            data = response.json()
            # Extract response text from various possible formats
            if isinstance(data, str):
                return data
            if isinstance(data, dict):
                return data.get("response", data.get("text", data.get("message", str(data))))
            return str(data)
    except Exception as e:
        log.error(f"❌ n8n webhook error ({url}): {e}")
        return "I'm having trouble processing that request right now."

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
# WEBSOCKET HANDLER
# =====================================================
@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):

    await ws.accept()

    user_id = "solomon_roth"
    recent_msgs = []
    processed_messages = set()

    # Conversation history for GPT context
    chat_history = []

    # Turn tracking
    turn_id = 0
    current_active_turn_id = 0

    calendar_kw = ["calendar", "meeting", "schedule", "appointment"]
    plate_kw = ["plate", "add", "to-do", "task", "notion", "list"]

    plate_add_kw = ["add", "put", "create", "new", "include"]
    plate_check_kw = ["what", "show", "see", "check", "read"]

    prompt = await get_notion_prompt()
    greet = prompt.splitlines()[0] if prompt else "Hello Solomon, I’m Silas."

    # GREETING
    try:
        log.info("👋 Sending greeting TTS")
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

    # =====================================================
    # Transcript processor — interruption + context
    # =====================================================
    async def transcript_processor():
        nonlocal recent_msgs, processed_messages, prompt, last_audio_time, turn_id, current_active_turn_id, chat_history
        try:
            while True:
                try:
                    transcript = await dg_queue.get()
                except asyncio.CancelledError:
                    break

                if not transcript:
                    continue

                log.info(f"📝 DG transcript (candidate): '{transcript}'")

                # Accept any transcript with at least one alphabetic character or digit
                # This ensures interruptions like "why?", "how?", "prevent it", "911", etc. become turns
                if not any(ch.isalnum() for ch in transcript):
                    log.info("⏭ Ignoring transcript without alphanumeric characters")
                    continue

                msg = transcript
                norm = _normalize(msg)
                
                # Less aggressive deduplication: only skip if seen in last 2 seconds
                # This allows quick re-statements after interrupts to go through
                now = time.time()
                recent_msgs = [(m, t) for (m, t) in recent_msgs if now - t < 2]
                if any(_is_similar(m, norm) for (m, t) in recent_msgs):
                    log.info(f"⏭ Skipping near-duplicate transcript: '{msg}'")
                    continue
                recent_msgs.append((norm, now))

                # Record user message in conversation history
                chat_history.append({"role": "user", "content": msg})

                # ============================================================
                # INTERRUPT/TURN-CANCELLATION LOGIC:
                # Each new transcript increments turn_id and sets it as the 
                # current_active_turn_id. This immediately supersedes any 
                # older turn still streaming GPT/TTS, forcing them to cancel.
                # ============================================================
                turn_id += 1
                current_turn = turn_id
                current_active_turn_id = current_turn  # supersede older streams
                log.info(f"🎯 NEW TURN {current_turn}: '{msg}' (history len={len(chat_history)})")

                # Context from mem0 + notion
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
                        log.info(f"⏭ Plate msg already processed: '{msg}'")
                        continue
                    processed_messages.add(msg)

                    reply = await send_to_n8n(N8N_PLATE_URL, msg)

                    # INTERRUPT CHECK: Before TTS generation
                    if current_turn != current_active_turn_id:
                        log.info(f"🔁 Plate turn {current_turn} abandoned (active={current_active_turn_id})")
                        continue

                    try:
                        log.info(f"🎙️ Plate TTS START turn={current_turn}")
                        tts = await openai_client.audio.speech.create(
                            model="gpt-4o-mini-tts",
                            voice="alloy",
                            input=reply
                        )
                        # INTERRUPT CHECK: After TTS generation but before sending
                        if current_turn != current_active_turn_id:
                            log.info(f"🔁 Plate turn {current_turn} abandoned after TTS (active={current_active_turn_id})")
                            continue
                        try:
                            await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                        except Exception:
                            pass
                        await ws.send_bytes(await tts.aread())
                        log.info(f"🎙️ Plate TTS SENT turn={current_turn}")
                    except Exception as e:
                        log.error(f"❌ TTS plate error: {e}")
                    continue

                # Calendar logic
                if any(k in lower for k in calendar_kw):
                    reply = await send_to_n8n(N8N_CALENDAR_URL, msg)

                    # INTERRUPT CHECK: Before TTS generation
                    if current_turn != current_active_turn_id:
                        log.info(f"🔁 Calendar turn {current_turn} abandoned (active={current_active_turn_id})")
                        continue

                    try:
                        log.info(f"🎙️ Calendar TTS START turn={current_turn}")
                        tts = await openai_client.audio.speech.create(
                            model="gpt-4o-mini-tts",
                            voice="alloy",
                            input=reply
                        )
                        # INTERRUPT CHECK: After TTS generation but before sending
                        if current_turn != current_active_turn_id:
                            log.info(f"🔁 Calendar turn {current_turn} abandoned after TTS (active={current_active_turn_id})")
                            continue
                        try:
                            await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                        except Exception:
                            pass
                        await ws.send_bytes(await tts.aread())
                        log.info(f"🎙️ Calendar TTS SENT turn={current_turn}")
                    except Exception as e:
                        log.error(f"❌ TTS calendar error: {e}")
                    continue

                # General GPT logic with conversation history
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
                        # INTERRUPT CHECK: If a new turn became active, cancel this stream
                        if current_turn != current_active_turn_id:
                            log.info(f"🔁 CANCEL STREAM turn={current_turn}, active={current_active_turn_id}")
                            break

                        delta = getattr(chunk.choices[0].delta, "content", "")
                        if not delta:
                            continue

                        assistant_full_text += delta
                        buffer += delta

                        if len(buffer) > CHUNK_CHAR_THRESHOLD:
                            # INTERRUPT CHECK: Before generating TTS
                            if current_turn != current_active_turn_id:
                                log.info(f"🔁 Turn {current_turn} cancelled before TTS chunk.")
                                break

                            try:
                                log.info(f"🎙️ TTS CHUNK START turn={current_turn}, len={len(buffer)}")
                                tts = await openai_client.audio.speech.create(
                                    model="gpt-4o-mini-tts",
                                    voice="alloy",
                                    input=buffer
                                )
                                # INTERRUPT CHECK: After TTS generation but before sending
                                if current_turn != current_active_turn_id:
                                    log.info(f"🔁 Turn {current_turn} cancelled after TTS chunk generation.")
                                    break
                                try:
                                    await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                                except Exception:
                                    pass
                                await ws.send_bytes(await tts.aread())
                                log.info(f"🎙️ TTS CHUNK SENT turn={current_turn}")
                            except Exception as e:
                                log.error(f"❌ TTS stream-chunk error: {e}")
                            buffer = ""

                    # Final buffer - send remaining text as TTS
                    if buffer.strip() and current_turn == current_active_turn_id:
                        try:
                            log.info(f"🎙️ TTS FINAL START turn={current_turn}, len={len(buffer.strip())}")
                            tts = await openai_client.audio.speech.create(
                                model="gpt-4o-mini-tts",
                                voice="alloy",
                                input=buffer
                            )
                            # INTERRUPT CHECK: Final verification before sending
                            if current_turn == current_active_turn_id:
                                try:
                                    await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                                except Exception:
                                    pass
                                await ws.send_bytes(await tts.aread())
                                log.info(f"🎙️ TTS FINAL SENT turn={current_turn}")
                        except Exception as e:
                            log.error(f"❌ TTS final-chunk error: {e}")

                    # Add assistant to history only if still active
                    if assistant_full_text.strip() and current_turn == current_active_turn_id:
                        chat_history.append({"role": "assistant", "content": assistant_full_text.strip()})
                        log.info(f"💾 Stored assistant turn {current_turn} in history (len={len(chat_history)})")

                    asyncio.create_task(mem0_add(user_id, msg))

                except Exception as e:
                    log.error(f"LLM error: {e}")

        except Exception as e:
            log.error(f"❌ transcript_processor fatal: {e}")

    transcript_task = asyncio.create_task(transcript_processor())

    # =====================================================
    # MAIN LOOP — browser audio -> Deepgram
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
                audio_bytes = audio_bytes + b"\x00"

            last_audio_time = time.time()

            # Minimal logging to reduce noise (no per-chunk PCM logging)

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
