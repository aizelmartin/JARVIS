"""
speech_engine.py
----------------
High-responsiveness, zero-lag Text-to-Speech (TTS) engine using pyttsx3.
Runs in a background worker thread.

Key Real-Time Optimizations:
  1. Zero Queue Backlog: Max queue size of 1. If a new sign arrives,
     any unstarted stale speech is immediately replaced.
  2. COM Initialization: Properly initializes COM apartment on Windows threads.
  3. Optimized Speaking Rate: Crisp 175 WPM pace for fast, snappy feedback.
  4. Non-blocking state check: `is_speaking` flag so callers know current audio state.
"""

import threading
import queue
import time
from typing import Optional


class SpeechEngine:
    """Snappy, real-time Text-to-Speech worker."""

    def __init__(self, rate: int = 175, volume: float = 1.0):
        self.rate = rate
        self.volume = volume

        # Queue size 1 ensures zero backlog buildup
        self._queue = queue.Queue(maxsize=1)
        self._running = True
        self._is_speaking = False
        self._last_spoken = ""
        self._last_spoken_time = 0.0

        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="SpeechWorker")
        self._thread.start()

    @property
    def is_speaking(self) -> bool:
        """Returns True if audio is actively playing."""
        return self._is_speaking

    def _worker_loop(self):
        """Worker thread loop with safe Windows COM initialization."""
        # Try COM initialization for Windows
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
        except Exception as e:
            print(f"[SpeechEngine] Warning: pyttsx3 init error: {e}")
            engine = None

        while self._running:
            try:
                text = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if text and engine and self._running:
                self._is_speaking = True
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as ex:
                    print(f"[SpeechEngine] Playback error: {ex}")
                finally:
                    self._is_speaking = False
                    self._last_spoken = text
                    self._last_spoken_time = time.time()

            self._queue.task_done()

        # Cleanup COM on shutdown
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass

    def speak(self, text: str, force: bool = False):
        """
        Enqueues text for immediate speech synthesis.
        Discards any stale pending speech in the queue so output is strictly real-time.
        """
        text = text.strip()
        if not text or text == "...":
            return

        # Empty the queue if a previous item is waiting so we never queue backlog
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except (queue.Empty, ValueError):
                break

        try:
            self._queue.put_nowait(text)
            print(f"[SpeechEngine] > Speaking: '{text}'")
        except queue.Full:
            pass

    def stop(self):
        """Stops the worker thread."""
        self._running = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except (queue.Empty, ValueError):
                break
