"""
speech_engine.py
----------------
High-performance, zero-latency Speech Engine for JARVIS.

Uses Windows native SAPI.SpVoice via COM with async execution flags:
  - SVSFlagsAsync (1): 100% non-blocking. Audio plays asynchronously in Windows
    native audio engine without freezing the camera feed or OpenCV window.
  - Never locks or goes silent on subsequent signs (unlike pyttsx3 event loop bugs).
  - Graceful pyttsx3 fallback on non-Windows platforms.
"""

import sys
import time
from typing import Optional


class SpeechEngine:
    """Native asynchronous Text-to-Speech engine."""

    SVS_FLAGS_ASYNC = 1               # SAPI async flag (non-blocking)
    SVS_PURGE_BEFORE_SPEAK = 2        # Purges previous sound to speak new sign immediately

    def __init__(self, rate: int = 2, volume: int = 100):
        """
        Args:
            rate   : Speed of speech (-10 to 10 for SAPI; 2 is brisk and clear).
            volume : Volume level (0 to 100).
        """
        self.rate = rate
        self.volume = volume
        self._sapi_voice = None
        self._pyttsx_engine = None
        self._last_spoken = ""
        self._last_spoken_time = 0.0

        self._init_engine()

    def _init_engine(self):
        """Initializes native Windows SAPI or falls back to pyttsx3."""
        if sys.platform == "win32":
            try:
                import win32com.client
                self._sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                self._sapi_voice.Rate = self.rate
                self._sapi_voice.Volume = self.volume
                print("[SpeechEngine] Initialized Windows Native SAPI.SpVoice (Async mode).")
                return
            except Exception as e:
                print(f"[SpeechEngine] Warning: SAPI init failed: {e}. Falling back to pyttsx3.")

        try:
            import pyttsx3
            self._pyttsx_engine = pyttsx3.init()
            self._pyttsx_engine.setProperty("rate", 175)
            self._pyttsx_engine.setProperty("volume", self.volume / 100.0)
            print("[SpeechEngine] Initialized pyttsx3 engine.")
        except Exception as ex:
            print(f"[SpeechEngine] Error: Could not initialize any TTS engine: {ex}")

    def speak(self, text: str, force: bool = False):
        """
        Speaks text asynchronously without blocking the video thread.
        """
        text = text.strip()
        if not text or text == "...":
            return

        now = time.time()
        # Cooldown check: don't re-speak the exact same word within 1.2s unless forced
        if not force and text == self._last_spoken and (now - self._last_spoken_time) < 1.2:
            return

        self._last_spoken = text
        self._last_spoken_time = now

        print(f"[SpeechEngine] 🔊 Speaking: '{text}'")

        if self._sapi_voice is not None:
            try:
                # SVS_FLAGS_ASYNC (1) speaks in background without blocking
                self._sapi_voice.Speak(text, self.SVS_FLAGS_ASYNC)
            except Exception as e:
                print(f"[SpeechEngine] SAPI speak error: {e}")
        elif self._pyttsx_engine is not None:
            try:
                self._pyttsx_engine.say(text)
                self._pyttsx_engine.runAndWait()
            except Exception as ex:
                print(f"[SpeechEngine] pyttsx3 speak error: {ex}")

    def stop(self):
        """Stops any currently playing audio."""
        if self._sapi_voice is not None:
            try:
                # Purge current speech buffer
                self._sapi_voice.Speak("", self.SVS_PURGE_BEFORE_SPEAK | self.SVS_FLAGS_ASYNC)
            except Exception:
                pass
