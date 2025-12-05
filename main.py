import os
import json
import logging
import asyncio
import time
from typing import List, Dict, Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
import uvicorn
import websockets
from asyncio import Queue

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("silas-main")

# =====================================================
# ENV
# =====================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MEMO_API_KEY = os.getenv("MEMO_API_KEY", "").strip()
NOTION_API_KEY = os.getenv("NOTION_API_KEY", "").strip()
NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID", "").strip()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()

N8N_CALENDAR_URL = os.getenv(
    "N8N_CALENDAR_URL",
    "https://n8n.marshall321.org/webhook/calendar-agent",
)
N8N_PLATE_URL = os.getenv(
    "N8N_PLATE_URL",
    "https://n8n.marshall321.org/webhook/agent/plate",
)

GPT_MODEL = os.getenv("SILAS_GPT_MODEL", "gpt-4o")
TTS_MODEL = os.getenv("SILAS_TTS_MODEL", "gpt-4o-mini-tts")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Size of text chunks we send to TTS
CHUNK_CHAR_THRESHOLD = 90  # ~short phrase/sentence

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Silas voice server running"}


@app.get("/health")
async def health():
    return {"ok": True}


# =====================================================
# MEM0 HELPERS
# =====================================================
async def mem0_search(user_id: str, query: str) -> List[Dict[str, Any]]:
    if not MEMO_API_KEY:
        return []
    headers = {"Authorization": f"Token {MEMO_API_KEY}"}
    payload = {"filters": {"user_id": user_id}, "query": query}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                "https://api.mem0.ai/v2/memories/",
                headers=headers,
                json=payload,
            )
        if r.status_code == 200:
            out = r.json()
            return out if isinstance(out, list) else []
    except Exception as e:
        log.error(f"MEM0 search error: {e}")
    return []


async def mem0_add(user_id: str, text: str) -> None:
    if not MEMO_API_KEY or not text:
        return
    headers = {"Authorization": f"Token {MEMO_API_KEY}"}
    payload = {
        "user_id": user_id,
        "messages": [{"role": "user", "content": text}],
    }
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(
                "https://api.mem0.ai/v1/memories/",
                headers=headers,
                json=payload,
            )
    except Exception as e:
        log.error(f"MEM0 add error: {e}")


def memory_context(memories: List[Dict[str, Any]]) -> str:
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
async def get_notion_prompt() -> str:
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
        parts: list[str] = []
        for blk in data.get("results", []):
            if blk.get("type") == "paragraph":
                rich = blk["paragraph"].get("rich_text", [])
                text = "".join(t.get("plain_text", "") for t in rich)
                if text:
                    parts.append(text)
        return "\n".join(parts).strip() or "You are Solomon Roth’s personal AI assistant, Silas."
    except Exception as e:
        log.error(f"❌ Notion error: {e}")
        return "You are Solomon Roth’s personal AI assistant, Silas."


@app.get("/prompt", response_class=PlainTextResponse)
async def prompt_endpoint():
    txt = await get_notion_prompt()
    return PlainTextResponse(txt, headers={"Access-Control-Allow-Origin": "*"})


# =====================================================
# HELPER: n8n calls (if you still use them)
# =====================================================
async def send_to_n8n(url: str, msg: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(url, json={"message": msg})
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return data.get("reply") or data.get("message") or json.dumps(data)
            return str(data)
        return f"Sorry, I had an error contacting that service (status {r.status_code})."
    except Exception as e:
        log.error(f"❌ n8n error calling {url}: {e}")
        return "Sorry, I had an error contacting that service."


# =====================================================
# CORE: Per-connection websocket handler
# =====================================================
@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    """
    Per-connection handler:
    - Accepts audio from browser (16-bit PCM @ 48kHz).
    - Forwards to Deepgram via one WS.
    - Receives transcripts via Deepgram.
    - For each transcript => new turn with GPT + TTS, using chat_history.
    - Uses turn_id/current_active_turn_id to cancel old turns when new ones arrive.
    """
    await ws.accept()
    user_id = "solomon_roth"  # hard-coded for now

    # Conversation state for THIS connection
    chat_history: list[dict[str, str]] = []
    turn_id: int = 0
    current_active_turn_id: int = 0

    # Load prompt once per connection
    base_prompt = await get_notion_prompt()
    greet = base_prompt.splitlines()[0] if base_prompt else "Hello Solomon, I’m Silas."

    # Greet once (turn_id=0)
    try:
        tts_greet = await openai_client.audio.speech.create(
            model=TTS_MODEL,
            voice="alloy",
            input=greet,
        )
        await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": 0}))
        await ws.send_bytes(await tts_greet.aread())
    except Exception as e:
        log.error(f"❌ Greeting TTS error: {e}")

    # =====================================================
    # Connect to Deepgram for this session
    # =====================================================
    if not DEEPGRAM_API_KEY:
        log.error("❌ No DEEPGRAM_API_KEY set")
        await ws.close(code=1011, reason="Missing Deepgram API key")
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
            ping_interval=20,   # periodic pings
            ping_timeout=20,
            max_size=None,
            close_timeout=5,
        )
        log.info("✅ Connected to Deepgram")
    except Exception as e:
        log.error(f"❌ Failed to connect to Deepgram WS: {e}")
        await ws.close(code=1011, reason="Failed to connect to Deepgram")
        return

    # Queue to move transcripts from DG listener to transcript processor
    dg_queue: Queue[str] = Queue()

    async def deepgram_listener():
        """Background task: read transcripts from Deepgram and enqueue them."""
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
                    # support nova-style and older results formats
                    if "channel" in data and isinstance(data["channel"], dict):
                        alts = data["channel"].get("alternatives", [])
                    elif "results" in data and isinstance(data["results"], dict):
                        chs = data["results"].get("channels", [])
                        if chs and isinstance(chs, list):
                            alts = chs[0].get("alternatives", [])
                        else:
                            alts = data["results"].get("alternatives", [])

                    transcript = ""
                    if alts and isinstance(alts, list):
                        transcript = alts[0].get("transcript", "").strip()

                    if transcript:
                        log.info(f"🧠 Deepgram transcript: {transcript}")
                        await dg_queue.put(transcript)

                except Exception as e:
                    log.error(f"❌ DG parse error: {e}")
        except websockets.exceptions.ConnectionClosedOK as e:
            log.warning(f"🔌 Deepgram connection closed normally: {e.code} {e.reason}")
        except websockets.exceptions.ConnectionClosedError as e:
            log.error(f"❌ Deepgram connection closed with error: {e.code} {e.reason}")
        except Exception as e:
            log.error(f"❌ DG listener fatal unexpected error: {e}")
        finally:
            # Close browser websocket for this session, but do not shut down app
            try:
                await ws.close(code=1011, reason="Deepgram connection closed")
            except Exception:
                pass

    dg_listener_task = asyncio.create_task(deepgram_listener())

    last_audio_time = time.time()

    async def dg_keepalive():
        nonlocal last_audio_time
        try:
            while True:
                await asyncio.sleep(1.2)
                if time.time() - last_audio_time > 1.5:
                    try:
                        silence = (b"\x00\x00") * 4800  # 100ms at 48kHz
                        await dg_ws.send(silence)
                        log.info("📨 Sent DG keepalive silence")
                    except Exception as e:
                        log.error(f"❌ Error sending keepalive to Deepgram: {e}")
                        break
        except asyncio.CancelledError:
            return

    keepalive_task = asyncio.create_task(dg_keepalive())

    # =====================================================
    # Transcript processor — turns + GPT/TTS + cancellation
    # =====================================================
    async def transcript_processor():
        nonlocal base_prompt, turn_id, current_active_turn_id, chat_history
        try:
            while True:
                try:
                    transcript = await dg_queue.get()
                except asyncio.CancelledError:
                    break

                if not transcript:
                    continue

                log.info(f"📝 DG transcript (candidate): '{transcript}'")

                # Very relaxed acceptance: any transcript with a letter
                if not any(ch.isalpha() for ch in transcript):
                    log.info("⏭ Ignoring transcript with no alphabetic chars")
                    continue

                msg = transcript

                # Append user message to chat history
                chat_history.append({"role": "user", "content": msg})

                # New turn
                turn_id += 1
                current_turn = turn_id
                current_active_turn_id = current_turn
                log.info(f"🎯 NEW TURN {current_turn}: '{msg}' (history len={len(chat_history)})")

                # Build system prompt with memories
                mems = await mem0_search(user_id, msg)
                ctx = memory_context(mems)
                sys_prompt = f"{base_prompt}\n\nFacts:\n{ctx}"
                system_msg = (
                    sys_prompt
                    + "\n\nSpeaking style: Respond concisely in 1–3 sentences, like live conversation. "
                      "Prioritize fast, direct answers over long explanations."
                )

                lower = msg.lower()

                # Special-case n8n calendar/plate if you want (optional)
                if any(k in lower for k in ["calendar", "meeting", "schedule", "appointment"]):
                    # Calendar
                    reply = await send_to_n8n(N8N_CALENDAR_URL, msg)
                    if current_turn != current_active_turn_id:
                        log.info(f"🔁 Calendar turn {current_turn} abandoned (active={current_active_turn_id})")
                        continue
                    try:
                        tts = await openai_client.audio.speech.create(
                            model=TTS_MODEL,
                            voice="alloy",
                            input=reply,
                        )
                        if current_turn != current_active_turn_id:
                            log.info(f"🔁 Calendar turn {current_turn} abandoned after TTS (active={current_active_turn_id})")
                            continue
                        await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                        await ws.send_bytes(await tts.aread())
                        log.info(f"🎙️ Calendar TTS SENT turn={current_turn}")
                    except Exception as e:
                        log.error(f"❌ TTS calendar error: {e}")
                    # You could optionally chat_history.append({"role": "assistant", "content": reply})
                    continue

                if any(k in lower for k in ["plate", "to-do", "task", "notion", "list"]):
                    # Plate / todo
                    reply = await send_to_n8n(N8N_PLATE_URL, msg)
                    if current_turn != current_active_turn_id:
                        log.info(f"🔁 Plate turn {current_turn} abandoned (active={current_active_turn_id})")
                        continue
                    try:
                        tts = await openai_client.audio.speech.create(
                            model=TTS_MODEL,
                            voice="alloy",
                            input=reply,
                        )
                        if current_turn != current_active_turn_id:
                            log.info(f"🔁 Plate turn {current_turn} abandoned after TTS (active={current_active_turn_id})")
                            continue
                        await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                        await ws.send_bytes(await tts.aread())
                        log.info(f"🎙️ Plate TTS SENT turn={current_turn}")
                    except Exception as e:
                        log.error(f"❌ TTS plate error: {e}")
                    continue

                # General GPT logic with chat_history
                try:
                    messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}] + chat_history
                    log.info(
                        f"🤖 GPT START turn={current_turn}, "
                        f"active={current_active_turn_id}, messages_len={len(messages)}"
                    )

                    stream = await openai_client.chat.completions.create(
                        model=GPT_MODEL,
                        messages=messages,
                        stream=True,
                    )

                    buffer = ""
                    assistant_full_text = ""

                    async for chunk in stream:
                        # If a newer turn became active while streaming, cancel
                        if current_turn != current_active_turn_id:
                            log.info(f"🔁 CANCEL STREAM turn={current_turn}, active={current_active_turn_id}")
                            break

                        delta = getattr(chunk.choices[0].delta, "content", "")
                        if not delta:
                            continue

                        assistant_full_text += delta
                        buffer += delta

                        if len(buffer) > CHUNK_CHAR_THRESHOLD:
                            if current_turn != current_active_turn_id:
                                log.info(f"🔁 Turn {current_turn} cancelled before TTS chunk.")
                                break

                            try:
                                log.info(f"🎙️ TTS CHUNK START turn={current_turn}, len={len(buffer)}")
                                tts = await openai_client.audio.speech.create(
                                    model=TTS_MODEL,
                                    voice="alloy",
                                    input=buffer,
                                )
                                if current_turn != current_active_turn_id:
                                    log.info(f"🔁 Turn {current_turn} cancelled after TTS chunk generation.")
                                    break
                                await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                                await ws.send_bytes(await tts.aread())
                                log.info(f"🎙️ TTS CHUNK SENT turn={current_turn}")
                            except Exception as e:
                                log.error(f"❌ TTS stream-chunk error: {e}")
                            buffer = ""

                    # Final TTS chunk, if any
                    if buffer.strip() and current_turn == current_active_turn_id:
                        try:
                            log.info(
                                f"🎙️ TTS FINAL START turn={current_turn}, "
                                f"len={len(buffer.strip())}"
                            )
                            tts = await openai_client.audio.speech.create(
                                model=TTS_MODEL,
                                voice="alloy",
                                input=buffer,
                            )
                            if current_turn == current_active_turn_id:
                                await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                                await ws.send_bytes(await tts.aread())
                                log.info(f"🎙️ TTS FINAL SENT turn={current_turn}")
                        except Exception as e:
                            log.error(f"❌ TTS final-chunk error: {e}")

                    # Only store assistant turn if this turn is still active
                    if assistant_full_text.strip() and current_turn == current_active_turn_id:
                        chat_history.append({"role": "assistant", "content": assistant_full_text.strip()})
                        log.info(
                            f"💾 Stored assistant turn {current_turn} in history "
                            f"(len={len(chat_history)})"
                        )

                    # Fire and forget mem0 add for the user message
                    asyncio.create_task(mem0_add(user_id, msg))

                except Exception as e:
                    log.error(f"LLM error: {e}")

        except Exception as e:
            log.error(f"❌ transcript_processor fatal: {e}")

    transcript_task = asyncio.create_task(transcript_processor())

    # =====================================================
    # MAIN LOOP: receive audio from browser and forward to Deepgram
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

            # Ensure even length (16-bit samples)
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes + b"\x00"

            last_audio_time = time.time()

            log.info(f"📡 PCM audio received — {len(audio_bytes)} bytes")

            try:
                await dg_ws.send(audio_bytes)
            except Exception as e:
                log.error(f"❌ Error sending audio to Deepgram WS: {e}")
                break

    except WebSocketDisconnect:
        pass
    finally:
        # Cleanup for this session ONLY
        try:
            keepalive_task.cancel()
        except Exception:
            pass
        try:
            transcript_task.cancel()
        except Exception:
            pass
        try:
            dg_listener_task.cancel()
        except Exception:
            pass
        try:
            await dg_ws.close()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass


# =====================================================
# SERVER START
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
