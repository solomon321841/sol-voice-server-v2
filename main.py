# =====================================================
# WEBSOCKET HANDLER - improved: single receiver + cancellable TTS tasks
# with Deepgram reconnect loop and exponential backoff
# =====================================================
@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):
    ...
    # Queues & task tracking
    incoming_audio_queue: Queue = Queue()
    dg_transcript_queue: Queue = Queue()
-    tts_tasks_by_turn: Dict[int, Set[asyncio.Task]] = {}
+    tts_tasks_by_turn: Dict[int, deque] = {}
+    tts_locks_by_turn: Dict[int, asyncio.Lock] = {}
+    MAX_TTS_TASKS_PER_TURN = 3
    last_audio_time = time.time()
...
    async def _tts_and_send(tts_text: str, t_turn: int):
-        try:
-            if USE_SSML:
-                tts_payload = make_ssml_from_text(tts_text, None)
-            else:
-                tts_payload = tts_text
-
-            if t_turn != current_active_turn_id:
-                log.info(f"🔁 TTS task for turn {t_turn} cancelled before create (active={current_active_turn_id})")
-                return
-
-            tts = await openai_client.audio.speech.create(model="gpt-4o-mini-tts", voice="cedar", input=tts_payload)
-
-            if t_turn != current_active_turn_id:
-                log.info(f"🔁 TTS task for turn {t_turn} cancelled after generation (active={current_active_turn_id})")
-                return
-
-            try:
-                await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": t_turn}))
-            except Exception:
-                pass
-
-            audio_bytes = await tts.aread()
-            if t_turn != current_active_turn_id:
-                log.info(f"🔁 TTS task for turn {t_turn} cancelled after aread (active={current_active_turn_id})")
-                return
-            try:
-                await ws.send_bytes(audio_bytes)
-                log.info(f"🎙️ TTS SENT for turn={t_turn}, len={len(audio_bytes)}")
-            except Exception as e:
-                log.error(f"Failed to send TTS bytes for turn {t_turn}: {e}")
-        except asyncio.CancelledError:
-            log.info(f"🔁 TTS task for turn {t_turn} cancelled (task cancelled).")
-            return
-        except Exception as e:
-            log.error(f"❌ TTS task error for turn {t_turn}: {e}")
+        try:
+            if t_turn != current_active_turn_id:
+                log.info(f"🔁 TTS task for turn {t_turn} cancelled before create (active={current_active_turn_id})")
+                return
+
+            lock = tts_locks_by_turn.setdefault(t_turn, asyncio.Lock())
+            async with lock:
+                if t_turn != current_active_turn_id:
+                    log.info(f"🔁 TTS task for turn {t_turn} cancelled before generation (active={current_active_turn_id})")
+                    return
+
+                tts_payload = make_ssml_from_text(tts_text, None) if USE_SSML else tts_text
+                tts = await openai_client.audio.speech.create(model="gpt-4o-mini-tts", voice="cedar", input=tts_payload)
+
+                if t_turn != current_active_turn_id:
+                    log.info(f"🔁 TTS task for turn {t_turn} cancelled after generation (active={current_active_turn_id})")
+                    return
+
+                try:
+                    await ws.send_text(json.dumps({"type": "tts_chunk", "turn_id": t_turn}))
+                except Exception:
+                    pass
+
+                audio_bytes = await tts.aread()
+                if t_turn != current_active_turn_id:
+                    log.info(f"🔁 TTS task for turn {t_turn} cancelled after aread (active={current_active_turn_id})")
+                    return
+                try:
+                    await ws.send_bytes(audio_bytes)
+                    log.info(f"🎙️ TTS SENT for turn={t_turn}, len={len(audio_bytes)}")
+                except Exception as e:
+                    log.error(f"Failed to send TTS bytes for turn {t_turn}: {e}")
+        except asyncio.CancelledError:
+            log.info(f"🔁 TTS task for turn {t_turn} cancelled (task cancelled).")
+            return
+        except Exception as e:
+            log.error(f"❌ TTS task error for turn {t_turn}: {e}")
+
+    def schedule_tts(tts_text: str, t_turn: int):
+        queue = tts_tasks_by_turn.setdefault(t_turn, deque())
+        while len(queue) >= MAX_TTS_TASKS_PER_TURN:
+            oldest = queue.popleft()
+            try:
+                oldest.cancel()
+            except Exception:
+                pass
+        task = asyncio.create_task(_tts_and_send(tts_text, t_turn))
+        queue.append(task)
+
+        def _cleanup(fut, t=t_turn):
+            q = tts_tasks_by_turn.get(t)
+            if q is not None:
+                try:
+                    q.remove(fut)
+                except ValueError:
+                    pass
+                if not q:
+                    tts_tasks_by_turn.pop(t, None)
+                    tts_locks_by_turn.pop(t, None)
+
+        task.add_done_callback(_cleanup)
+        return task
...
-                    t_task = asyncio.create_task(_tts_and_send(reply, current_turn))
-                    tts_tasks_by_turn.setdefault(current_turn, set()).add(t_task)
-                    t_task.add_done_callback(lambda fut, t=current_turn: tts_tasks_by_turn.get(t, set()).discard(fut))
+                    schedule_tts(reply, current_turn)
                     continue
...
-                    t_task = asyncio.create_task(_tts_and_send(reply, current_turn))
-                    tts_tasks_by_turn.setdefault(current_turn, set()).add(t_task)
-                    t_task.add_done_callback(lambda fut, t=current_turn: tts_tasks_by_turn.get(t, set()).discard(fut))
+                    schedule_tts(reply, current_turn)
                     continue
...
-                        if len(buffer) >= 22 or buffer.endswith(". ") or buffer.endswith("?") or buffer.endswith("! ") or buffer.endswith(","):
-                            chunk_text = buffer
-                            buffer = ""
-                            t_task = asyncio.create_task(_tts_and_send(chunk_text, current_turn))
-                            tts_tasks_by_turn.setdefault(current_turn, set()).add(t_task)
-                            t_task.add_done_callback(lambda fut, t=current_turn: tts_tasks_by_turn.get(t, set()).discard(fut))
-
-                    if buffer.strip() and current_turn == current_active_turn_id:
-                        t_task = asyncio.create_task(_tts_and_send(buffer, current_turn))
-                        tts_tasks_by_turn.setdefault(current_turn, set()).add(t_task)
-                        t_task.add_done_callback(lambda fut, t=current_turn: tts_tasks_by_turn.get(t, set()).discard(fut))
+                        if (
+                            len(buffer) >= 70
+                            or buffer.endswith((". ", "? ", "! ", ".", "?", "!"))
+                        ):
+                            chunk_text = buffer
+                            buffer = ""
+                            schedule_tts(chunk_text, current_turn)
+
+                    if buffer.strip() and current_turn == current_active_turn_id:
+                        schedule_tts(buffer, current_turn)
...
-                    for t_id, tasks in list(tts_tasks_by_turn.items()):
-                        if t_id != current_active_turn_id:
-                            for t in tasks:
-                                try:
-                                    t.cancel()
-                                except Exception:
-                                    pass
-                            tts_tasks_by_turn.pop(t_id, None)
+                    for t_id, tasks in list(tts_tasks_by_turn.items()):
+                        if t_id != current_active_turn_id:
+                            for t in list(tasks):
+                                try:
+                                    t.cancel()
+                                except Exception:
+                                    pass
+                            tts_tasks_by_turn.pop(t_id, None)
+                            tts_locks_by_turn.pop(t_id, None)
...
-        for tasks in tts_tasks_by_turn.values():
-            for t in tasks:
-                try:
-                    t.cancel()
-                except Exception:
-                    pass
+        for tasks in tts_tasks_by_turn.values():
+            for t in list(tasks):
+                try:
+                    t.cancel()
+                except Exception:
+                    pass
+        tts_tasks_by_turn.clear()
+        tts_locks_by_turn.clear()
