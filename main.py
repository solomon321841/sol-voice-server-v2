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
# n8n ENDPOINTS
# =====================================================
N8N_CALENDAR_URL = "https://n8n.marshall321.org/webhook/calendar-agent"
N8N_PLATE_URL = "https://n8n.marshall321.org/webhook/agent/plate"

# =====================================================
# MODEL
# =====================================================
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = "gpt-5.1"
CHUNK_CHAR_THRESHOLD = 90  # for natural speech chunks

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
# WEBSOCKET HANDLER
# =====================================================
@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):
  await ws.accept()
  user_id = "solomon_roth"

  # Connection state tracking
  ws_connected = True
  dg_connected = False

  # Conversation history
  chat_history = []

  # Turn tracking
  turn_id = 0
  current_active_turn_id = 0

  calendar_kw = ["calendar", "meeting", "schedule", "appointment"]
  plate_kw = ["plate", "add", "to-do", "task", "notion", "list"]

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
      if ws_connected:
          await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": 0}))
          await ws.send_bytes(await tts_greet.aread())
  except Exception as e:
      log.error(f"❌ Greeting TTS error: {e}")
      ws_connected = False

  # =====================================================
  # Connect to Deepgram with auto-ping
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
          ping_interval=20,
          ping_timeout=20,
          max_size=None,
          close_timeout=5
      )
      dg_connected = True
      log.info("✅ Connected to Deepgram")
  except Exception as e:
      log.error(f"❌ Failed to connect to Deepgram WS: {e}")
      ws_connected = False
      await ws.close(code=1011, reason="Failed to connect to Deepgram")
      return

  dg_queue = Queue()

  async def deepgram_listener_task():
      nonlocal dg_connected, ws_connected
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
          dg_connected = False
          # Only close the browser websocket if it's still connected
          if ws_connected:
              try:
                  ws_connected = False
                  await ws.close(code=1011, reason="Deepgram connection closed")
              except Exception as close_err:
                  log.error(f"❌ Error closing browser websocket: {close_err}")

  asyncio.create_task(deepgram_listener_task())

  last_audio_time = time.time()

  async def dg_keepalive_task():
      nonlocal last_audio_time, dg_connected
      try:
          while dg_connected:
              await asyncio.sleep(1.2)
              if time.time() - last_audio_time > 1.5:
                  try:
                      silence = (b"\x00\x00") * 4800
                      await dg_ws.send(silence)
                      log.debug("📨 Sent DG keepalive silence")
                  except Exception as e:
                      log.error(f"❌ Error sending keepalive to Deepgram: {e}")
                      dg_connected = False
                      break
      except asyncio.CancelledError:
          return

  keepalive_task = asyncio.create_task(dg_keepalive_task())

  # =====================================================
  # Transcript processor — interruption + context
  # =====================================================
  async def transcript_processor():
      nonlocal prompt, last_audio_time, turn_id, current_active_turn_id, chat_history, ws_connected
      try:
          while True:
              try:
                  transcript = await dg_queue.get()
              except asyncio.CancelledError:
                  break

              if not transcript:
                  continue

              log.info(f"📝 DG transcript (candidate): '{transcript}'")

              # Require at least 2 words to reduce false positives from noise
              words = transcript.split()
              if len(words) < 2:
                  log.info(f"⏭ Ignoring short transcript (< 2 words): '{transcript}'")
                  continue

              msg = transcript

              # Add to conversation history
              chat_history.append({"role": "user", "content": msg})

              # New turn
              turn_id += 1
              current_turn = turn_id
              current_active_turn_id = current_turn
              log.info(f"🎯 NEW TURN {current_turn}: '{msg}' (history len={len(chat_history)})")

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

              # General GPT logic with chat_history
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
                      # Check if this turn was interrupted
                      if current_turn != current_active_turn_id:
                          log.info(f"🔁 CANCEL STREAM turn={current_turn}, active={current_active_turn_id}")
                          break

                      delta = getattr(chunk.choices[0].delta, "content", "")
                      if not delta:
                          continue

                      assistant_full_text += delta
                      buffer += delta

                      if len(buffer) > CHUNK_CHAR_THRESHOLD:
                          # Double-check turn is still active before TTS
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
                              # Final check before sending
                              if current_turn != current_active_turn_id:
                                  log.info(f"🔁 Turn {current_turn} cancelled after TTS chunk generation.")
                                  break
                              # Check websocket is still connected before sending
                              if ws_connected:
                                  try:
                                      await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                                      await ws.send_bytes(await tts.aread())
                                      log.info(f"🎙️ TTS CHUNK SENT turn={current_turn}")
                                  except Exception as send_err:
                                      log.error(f"❌ Error sending TTS chunk: {send_err}")
                                      ws_connected = False
                                      break
                              else:
                                  log.info(f"⏭ WebSocket closed, skipping TTS for turn {current_turn}")
                                  break
                          except Exception as e:
                              log.error(f"❌ TTS stream-chunk error: {e}")
                          buffer = ""

                  # Send final buffer if turn is still active
                  if buffer.strip() and current_turn == current_active_turn_id:
                      try:
                          log.info(f"🎙️ TTS FINAL START turn={current_turn}, len={len(buffer.strip())}")
                          tts = await openai_client.audio.speech.create(
                              model="gpt-4o-mini-tts",
                              voice="alloy",
                              input=buffer
                          )
                          if current_turn == current_active_turn_id and ws_connected:
                              try:
                                  await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": current_turn}))
                                  await ws.send_bytes(await tts.aread())
                                  log.info(f"🎙️ TTS FINAL SENT turn={current_turn}")
                              except Exception as send_err:
                                  log.error(f"❌ Error sending final TTS: {send_err}")
                                  ws_connected = False
                      except Exception as e:
                          log.error(f"❌ TTS final-chunk error: {e}")

                  # Only append to chat history if turn completed successfully
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
      while ws_connected and dg_connected:
          try:
              audio_bytes = await ws.receive_bytes()
          except WebSocketDisconnect:
              log.info("🔌 Browser websocket disconnected")
              ws_connected = False
              break
          except Exception as e:
              log.error(f"❌ WebSocket receive error: {e}")
              ws_connected = False
              break

          if not audio_bytes:
              continue

          # Ensure even byte count for 16-bit PCM
          if len(audio_bytes) % 2 != 0:
              audio_bytes = audio_bytes + b"\x00"

          last_audio_time = time.time()

          # Log audio reception without verbose PCM sample data
          log.debug(f"📡 PCM audio received — {len(audio_bytes)} bytes")

          try:
              await dg_ws.send(audio_bytes)
          except Exception as e:
              log.error(f"❌ Error sending audio to Deepgram WS: {e}")
              dg_connected = False
              break

  except WebSocketDisconnect:
      ws_connected = False
      log.info("🔌 Browser websocket disconnected (outer)")
  except Exception as outer_err:
      log.error(f"❌ Main loop error: {outer_err}")
      ws_connected = False
  finally:
      log.info("🧹 Cleaning up websocket session...")
      ws_connected = False
      dg_connected = False
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
