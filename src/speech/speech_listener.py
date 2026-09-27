"""
speech_listener.py
------------------
Asynchronous Speech-to-Text (STT) listener using sounddevice and SpeechRecognition.
Enables two-way communication: a hearing person can speak into the microphone,
and their words are transcribed onto the screen for the deaf/mute user.
"""

import threading
import time
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from typing import Optional


class SpeechListener:
    """Non-blocking background Speech-to-Text listener."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.recognizer = sr.Recognizer()
        self.is_listening = False
        self.last_text = ""
        self.status_msg = "Press M to talk"
        self._thread: Optional[threading.Thread] = None

    def listen_async(self, duration_sec: float = 3.5):
        """Starts recording and transcribing in a background daemon thread."""
        if self.is_listening:
            return

        self.is_listening = True
        self.status_msg = "🎙️ Listening... speak now"
        self._thread = threading.Thread(
            target=self._record_and_transcribe,
            args=(duration_sec,),
            daemon=True,
            name="STTWorker",
        )
        self._thread.start()

    def _record_and_transcribe(self, duration_sec: float):
        try:
            num_samples = int(duration_sec * self.sample_rate)
            # Record from default microphone
            audio_buffer = sd.rec(
                num_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()

            self.status_msg = "Transcribing speech..."
            raw_bytes = audio_buffer.tobytes()
            audio_data = sr.AudioData(raw_bytes, self.sample_rate, 2)

            # Recognize speech
            try:
                text = self.recognizer.recognize_google(audio_data)
                self.last_text = text
                self.status_msg = f"Heard: '{text}'"
                print(f"[SpeechListener] 🎙️ Transcribed: '{text}'")
            except sr.UnknownValueError:
                self.status_msg = "Could not understand audio"
            except sr.RequestError as e:
                self.status_msg = "Speech service error (check internet)"
        except Exception as ex:
            self.status_msg = f"Mic error: {ex}"
            print(f"[SpeechListener] Error: {ex}")
        finally:
            self.is_listening = False

    def clear(self):
        """Clears the current transcription."""
        self.last_text = ""
        self.status_msg = "Press M to talk"
