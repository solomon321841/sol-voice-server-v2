import os
import json
import logging
import asyncio
import time
import struct
from typing import List, Dict, Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from openai import AsyncOpenAI

from asyncio import Queue

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("main")

# =====================================================
# ENV
# =====================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MEMO_API_KEY = os.getenv("MEMO_API_KEY", "").strip()
NOTION_API_KEY = os.getenv("NOTION_API_KEY", "").strip()
NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID", "").strip()

N8N_CALENDAR_URL = "https://n8n.marshall321.org/webhook/calendar-agent"
N8N_PLATE_URL = "https://n8n.marshall321.org/webhook/agent/plate"

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

GPT_MODEL = "gpt-5.1"
TTS_MODEL = "gpt-4o-mini-tts"
ASR_MODEL = "whisper-1"  # or "gpt-4o-mini-transcribe" if your account has it

CHUNK_CHAR_THRESHOLD = 90

SAMPLE_RATE = 48000
BYTES_PER_SAMPLE = 2
NUM_CHANNELS = 1

MAX_BUFFER_SECONDS = 6.0        # keep last N seconds of audio
ASR_WINDOW_SECONDS = 1.0        # each transcription window size
ASR_INTERVAL_SECONDS = 0.7      # how often we call ASR

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
# MEM0 HELPERS
# =====================================================
async def mem0_search(user_id: str, query: str):
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


async def mem0_add(user_id: str, text: str):
    if not MEMO_API_KEY or not text:
        return
    headers = {"Authorization": f"Token {MEMO_API_KEY}"}
    payload = {"user_id": user_id, "messages": [{"role": "user", "content": text}]}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(
                "https://api.mem0.ai/v1/memories/",
                headers=headers,
                json=payload,
            )
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
                    parts.append(
                        "".join(
                            [t.get("plain_text", "") for t in blk["paragraph"]["rich_text"]]
                        )
                    )
            return (
                "\n".join(parts).strip()
                or "You are Solomon Roth’s AI assistant, Silas."
            )
    except Exception as e:
        log.error(f"❌ Notion error: {e}")
        return "You are Solomon Roth’s AI assistant, Silas."


@app.get("/prompt", response_class=PlainTextResponse)
async def get_prompt_text():
    txt = await get_notion_prompt()
    return PlainTextResponse(txt, headers={"Access-Control-Allow-Origin": "*"})


# =====================================================
# n8n helper
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
        return f"Error contacting service (status {r.status_code})."
    except Exception as e:
        log.error(f"❌ n8n error calling {url}: {e}")
        return "Sorry, I had an error contacting that service."


# =====================================================
# WAV WRAPPER FOR PCM
# =====================================================
def pcm_to_wav_bytes(pcm: bytes, sample_rate: int, num_channels: int) -> bytes:
    """
    Wrap raw PCM16 (little-endian) into a minimal WAV container.
    """
    num_samples = len(pcm) // BYTES_PER_SAMPLE
    byte_rate = sample_rate * num_channels * BYTES_PER_SAMPLE
    block_align = num_channels * BYTES_PER_SAMPLE
    bits_per_sample = BYTES_PER_SAMPLE * 8
    subchunk2_size = len(pcm)
    chunk_size = 36 + subchunk2_size

    # RIFF header
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,              # PCM
        1,               # AudioFormat = 1 (PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        subchunk2_size,
    )
    return header + pcm


# =====================================================
# Simple incremental diff
# =====================================================
def incremental_new_text(old: str, new: str) -> Optional[str]:
    """
    Return the newly added suffix when ASR partials grow over time.
    If new is shorter or same, return None.
    """
    if not new or len(new) <= len(old):
        return None
    if new.startswith(old):
        return new[len(old):].strip()
    # If not a pure prefix, just return the whole thing
    return new.strip()


# =====================================================
# WEBSOCKET HANDLER (OpenAI ASR instead of Deepgram)
# =====================================================
@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):
    await ws.accept()
    user_id = "solomon_roth"

    # Per-connection state
    chat_history: list[dict[str, str]] = []
    turn_id = 0
    current_active_turn_id = 0

    calendar_kw = ["calendar", "meeting", "schedule", "appointment"]
    plate_kw = ["plate", "add", "to-do", "task", "notion", "list"]

    prompt = await get_notion_prompt()
    greet = prompt.splitlines()[0] if prompt else "Hello Solomon, I’m Silas."

    # Greeting (turn 0)
    try:
        log.info("👋 Sending greeting TTS")
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
    # Audio buffer for ASR
    # =====================================================
    max_buffer_bytes = int(MAX_BUFFER_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE)
    asr_window_bytes = int(ASR_WINDOW_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE)

    audio_buffer = bytearray()
    last_asr_text: str = ""
    last_asr_time = 0.0

    asr_queue: Queue[str] = Queue()

    # =====================================================
    # ASR worker: periodically transcribe latest buffer slice
    # =====================================================
    async def asr_worker():
        nonlocal last_asr_text, last_asr_time, audio_buffer
        try:
            while True:
                await asyncio.sleep(ASR_INTERVAL_SECONDS)
                if not audio_buffer:
                    continue

                now = time.time()
                if now - last_asr_time < ASR_INTERVAL_SECONDS * 0.5:
                    continue
                last_asr_time = now

                # Take the last `asr_window_bytes` from buffer
                if len(audio_buffer) <= asr_window_bytes:
                    window_pcm = bytes(audio_buffer)
                else:
                    window_pcm = bytes(audio_buffer[-asr_window_bytes:])

                if not window_pcm:
                    continue

                # Wrap PCM into WAV
                wav_bytes = pcm_to_wav_bytes(
                    window_pcm, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS
                )

                try:
                    log.info(f"🎧 ASR call with {len(window_pcm)} pcm bytes -> {len(wav_bytes)} wav bytes")
                    transcript = await openai_client.audio.transcriptions.create(
                        model=ASR_MODEL,
                        file=("audio.wav", wav_bytes, "audio/wav"),
                    )
                    text = getattr(transcript, "text", "") or ""
                    text = text.strip()
                    if not text:
                        continue

                    log.info(f"🧠 ASR full text: '{text}'")
                    new_part = incremental_new_text(last_asr_text, text)
                    if new_part:
                        log.info(f"🧠 ASR new segment: '{new_part}'")
                        last_asr_text = text
                        await asr_queue.put(new_part)
                except Exception as e:
                    log.error(f"❌ ASR error: {e}")
                    continue

        except asyncio.CancelledError:
            return
        except Exception as e:
            log.error(f"❌ asr_worker fatal: {e}")

    asr_task = asyncio.create_task(asr_worker())

    # =====================================================
    # Transcript processor: ASR text -> turns -> GPT + TTS
    # =====================================================
    async def transcript_processor():
        nonlocal prompt, turn_id, current_active_turn_id, chat_history
        try:
            while True:
                try:
                    segment = await asr_queue.get()
                except asyncio.CancelledError:
                    break

                if not segment:
                    continue

                msg = segment.strip()
                if not any(ch.isalpha() for ch in msg):
                    continue

                log.info(f"📝 ASR segment (candidate): '{msg}'")

                # Append user message
                chat_history.append({"role": "user", "content": msg})

                # New turn
                turn_id += 1
                current_turn = turn_id
                current_active_turn_id = current_turn
                log.info(
                    f"🎯 NEW TURN {current_turn}: '{msg}' "
                    f"(history len={len(chat_history)})"
                )

                # Context
                mems = await mem0_search(user_id, msg)
                ctx = memory_context(mems)
                sys_prompt = f"{prompt}\n\nFacts:\n{ctx}"
                system_msg = (
                    sys_prompt
                    + "\n\nSpeaking style: Respond concisely in 1–3 sentences, like live conversation. "
                      "Prioritize fast, direct answers over long explanations."
                )

                lower = msg.lower()

                # Plate
                if any(k in lower for k in plate_kw):
                    reply = await send_to_n8n(N8N_PLATE_URL, msg)
                    if current_turn != current_active_turn_id:
                        log.info(
                            f"🔁 Plate turn {current_turn} abandoned "
                            f"(active={current_active_turn_id})"
                        )
                        continue
                    try:
                        tts = await openai_client.audio.speech.create(
                            model=TTS_MODEL,
                            voice="alloy",
                            input=reply,
                        )
                        if current_turn != current_active_turn_id:
                            log.info(
                                f"🔁 Plate turn {current_turn} abandoned after TTS "
                                f"(active={current_active_turn_id})"
                            )
                            continue
                        await ws.send_text(
                            json.dumps(
                                {"type": "tts_chunk", "turn_id": current_turn}
                            )
                        )
                        await ws.send_bytes(await tts.aread())
                        log.info(f"🎙️ Plate TTS SENT turn={current_turn}")
                    except Exception as e:
                        log.error(f"❌ TTS plate error: {e}")
                    continue

                # Calendar
                if any(k in lower for k in calendar_kw):
                    reply = await send_to_n8n(N8N_CALENDAR_URL, msg)
                    if current_turn != current_active_turn_id:
                        log.info(
                            f"🔁 Calendar turn {current_turn} abandoned "
                            f"(active={current_active_turn_id})"
                        )
                        continue
                    try:
                        tts = await openai_client.audio.speech.create(
                            model=TTS_MODEL,
                            voice="alloy",
                            input=reply,
                        )
                        if current_turn != current_active_turn_id:
                            log.info(
                                f"🔁 Calendar turn {current_turn} abandoned after TTS "
                                f"(active={current_active_turn_id})"
                            )
                            continue
                        await ws.send_text(
                            json.dumps(
                                {"type": "tts_chunk", "turn_id": current_turn}
                            )
                        )
                        await ws.send_bytes(await tts.aread())
                        log.info(f"🎙️ Calendar TTS SENT turn={current_turn}")
                    except Exception as e:
                        log.error(f"❌ TTS calendar error: {e}")
                    continue

                # GPT + TTS
                try:
                    messages = [{"role": "system", "content": system_msg}] + chat_history
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
                        if current_turn != current_active_turn_id:
                            log.info(
                                f"🔁 CANCEL STREAM turn={current_turn}, "
                                f"active={current_active_turn_id}"
                            )
                            break

                        delta = getattr(chunk.choices[0].delta, "content", "")
                        if not delta:
                            continue

                        assistant_full_text += delta
                        buffer += delta

                        if len(buffer) > CHUNK_CHAR_THRESHOLD:
                            if current_turn != current_active_turn_id:
                                log.info(
                                    f"🔁 Turn {current_turn} cancelled "
                                    f"before TTS chunk."
                                )
                                break
                            try:
                                tts = await openai_client.audio.speech.create(
                                    model=TTS_MODEL,
                                    voice="alloy",
                                    input=buffer,
                                )
                                if current_turn != current_active_turn_id:
                                    log.info(
                                        f"🔁 Turn {current_turn} cancelled "
                                        f"after TTS chunk generation."
                                    )
                                    break
                                await ws.send_text(
                                    json.dumps(
                                        {"type": "tts_chunk", "turn_id": current_turn}
                                    )
                                )
                                await ws.send_bytes(await tts.aread())
                                log.info(f"🎙️ TTS CHUNK SENT turn={current_turn}")
                            except Exception as e:
                                log.error(f"❌ TTS stream-chunk error: {e}")
                            buffer = ""

                    # Final bit
                    if buffer.strip() and current_turn == current_active_turn_id:
                        try:
                            tts = await openai_client.audio.speech.create(
                                model=TTS_MODEL,
                                voice="alloy",
                                input=buffer,
                            )
                            if current_turn == current_active_turn_id:
                                await ws.send_text(
                                    json.dumps(
                                        {"type": "tts_chunk", "turn_id": current_turn}
                                    )
                                )
                                await ws.send_bytes(await tts.aread())
                                log.info(f"🎙️ TTS FINAL SENT turn={current_turn}")
                        except Exception as e:
                            log.error(f"❌ TTS final-chunk error: {e}")

                    if assistant_full_text.strip() and current_turn == current_active_turn_id:
                        chat_history.append(
                            {"role": "assistant", "content": assistant_full_text.strip()}
                        )
                        log.info(
                            f"💾 Stored assistant turn {current_turn} in history "
                            f"(len={len(chat_history)})"
                        )

                    asyncio.create_task(mem0_add(user_id, msg))

                except Exception as e:
                    log.error(f"LLM error: {e}")

        except Exception as e:
            log.error(f"❌ transcript_processor fatal: {e}")

    transcript_task = asyncio.create_task(transcript_processor())

    # =====================================================
    # MAIN LOOP: browser audio -> ASR buffer
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

            # Ensure even length
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes + b"\x00"

            # Append to rolling buffer
            audio_buffer.extend(audio_bytes)
            if len(audio_buffer) > max_buffer_bytes:
                # Keep only last N seconds
                excess = len(audio_buffer) - max_buffer_bytes
                del audio_buffer[:excess]

            log.info(
                f"📡 PCM audio received — {len(audio_bytes)} bytes "
                f"(buffer={len(audio_buffer)})"
            )

    except WebSocketDisconnect:
        pass
    finally:
        try:
            asr_task.cancel()
        except Exception:
            pass
        try:
            transcript_task.cancel()
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
