"""
speech_engine.py
----------------
Non-blocking Text-to-Speech (TTS) engine using pyttsx3.
Runs in a background worker thread so the camera stream and OpenCV
loop never lag or freeze when speaking phrases.
"""

import threading
import queue
import time
from typing import Optional


class SpeechEngine:
    """
    Asynchronous Text-to-Speech worker.
    Queues spoken phrases and speaks them in a background daemon thread.
    Includes cooldown to avoid repeating the same sign consecutively.
    """

    def __init__(self, rate: int = 160, volume: float = 1.0, repeat_cooldown: float = 3.0):
        self.rate = rate
        self.volume = volume
        self.repeat_cooldown = repeat_cooldown

        self._queue = queue.Queue()
        self._last_spoken_text = ""
        self._last_spoken_time = 0.0
        self._running = True

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self):
        """Worker thread running pyttsx3 engine."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
        except Exception as e:
            print(f"[SpeechEngine] Warning: Could not initialize pyttsx3: {e}")
            engine = None

        while self._running:
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text and engine:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as ex:
                    print(f"[SpeechEngine] Speech playback error: {ex}")
            self._queue.task_done()

    def speak(self, text: str, force: bool = False):
        """
        Enqueues text for speech synthesis if it differs from the recent text
        or if the repeat cooldown has elapsed.
        """
        text = text.strip()
        if not text or text == "...":
            return

        now = time.time()
        if not force and text == self._last_spoken_text and (now - self._last_spoken_time) < self.repeat_cooldown:
            return

        self._last_spoken_text = text
        self._last_spoken_time = now
        self._queue.put(text)
        print(f"[SpeechEngine] Speaking: '{text}'")

    def stop(self):
        """Stops the worker thread."""
        self._running = False
