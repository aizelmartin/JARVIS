"""
main.py — JARVIS Universal Sign Language Translator
===================================================
Complete Two-Way Communication System:
  1. Sign Language -> Speech:
     - 640x480 video feed with dual-hand MediaPipe tracking
     - 126-feature normalized landmark vector
     - Real-time ML classifier with confidence thresholding
     - Hold-to-Confirm dwell stabilization (0.3s)
     - Sentence Builder: Accumulates signs into full phrases
     - Native Windows SAPI asynchronous Text-to-Speech (TTS)
  2. Speech -> Text:
     - Microphone listening via sounddevice + SpeechRecognition
     - Displays hearing person's spoken words on screen for deaf/mute user

Keyboard Controls:
  Q         — Quit
  T         — Toggle Voice (TTS) ON / OFF
  ENTER     — Speak the entire constructed sentence aloud
  BACKSPACE — Delete the last added word from sentence
  C         — Clear the constructed sentence
  M         — Listen to microphone (Speech-to-Text for 3.5s)
  S         — Force speak current sign immediately
"""

import cv2
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.camera.camera_stream import CameraStream
from src.hand_tracking.hand_detector import HandDetector
from src.ml.classifier import SignClassifier
from src.ml.sentence_builder import SentenceBuilder
from src.speech.speech_engine import SpeechEngine
from src.speech.speech_listener import SpeechListener

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "saved_models", "sign_classifier.pkl")


def draw_hud(
    frame,
    sign: str,
    conf: float,
    n_hands: int,
    hands_lbl: list,
    fps: float,
    tts_enabled: bool,
    model_name: str,
    hold_ratio: float,
    sentence: str,
    stt_status: str,
    stt_text: str,
    is_listening: bool,
):
    """Renders a sleek, comprehensive two-way communication HUD."""
    h, w = frame.shape[:2]

    # ── Top Bar: Telemetry & Status ──────────────────────────────────── #
    cv2.rectangle(frame, (0, 0), (w, 38), (18, 18, 18), -1)
    cv2.line(frame, (0, 38), (w, 38), (60, 60, 60), 1)

    cv2.putText(frame, "JARVIS", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 230, 130), 2, cv2.LINE_AA)
    cv2.putText(frame, f"| FPS: {fps:.1f}", (105, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"| ML: {model_name}", (195, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 240), 1, cv2.LINE_AA)

    # Hand Mode
    mode_text = f"{n_hands} HAND(S)" if n_hands > 0 else "IDLE"
    mode_color = (0, 220, 120) if n_hands == 2 else ((255, 180, 0) if n_hands == 1 else (110, 110, 110))
    cv2.putText(frame, mode_text, (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, mode_color, 2, cv2.LINE_AA)

    # ── Middle-Top: Speech-to-Text (Hearing Person Response) ─────────── #
    stt_banner_h = 32
    stt_y = 42
    stt_bg = (35, 25, 10) if is_listening else (25, 25, 25)
    cv2.rectangle(frame, (0, stt_y), (w, stt_y + stt_banner_h), stt_bg, -1)
    cv2.line(frame, (0, stt_y + stt_banner_h), (w, stt_y + stt_banner_h), (50, 50, 50), 1)

    mic_icon = "MIC: [RECORDING...]" if is_listening else "MIC: [IDLE - Press M]"
    mic_color = (0, 140, 255) if is_listening else (140, 140, 140)
    cv2.putText(frame, mic_icon, (12, stt_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mic_color, 1, cv2.LINE_AA)

    display_stt = f"Heard: \"{stt_text}\"" if stt_text else stt_status
    stt_text_color = (0, 240, 255) if stt_text else (160, 160, 160)
    cv2.putText(frame, display_stt, (180, stt_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, stt_text_color, 1, cv2.LINE_AA)

    # ── Bottom Section: Sign Translation & Sentence Construction ─────── #
    card_h = 135
    cv2.rectangle(frame, (0, h - card_h), (w, h), (16, 16, 16), -1)
    cv2.line(frame, (0, h - card_h), (w, h - card_h), (60, 60, 60), 1)

    # Row 1: Live Sign & Confidence
    has_sign = (sign != "..." and conf >= 0.60)
    display_sign = sign.upper() if has_sign else "..."
    sign_color = (0, 240, 140) if has_sign else (110, 110, 110)

    cv2.putText(frame, "SIGN:", (14, h - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, display_sign, (65, h - 103), cv2.FONT_HERSHEY_SIMPLEX, 0.85, sign_color, 2, cv2.LINE_AA)

    # Confidence & Hold Bar
    if has_sign:
        bar_x, bar_y = 230, h - 116
        bar_w = int(110 * conf)
        cv2.putText(frame, f"CONF: {conf*100:.0f}%", (bar_x, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x + 75, bar_y), (bar_x + 185, bar_y + 12), (45, 45, 45), -1)
        cv2.rectangle(frame, (bar_x + 75, bar_y), (bar_x + 75 + bar_w, bar_y + 12), (0, 210, 120), -1)

        # Hold stability progress
        hold_w = int(110 * hold_ratio)
        hold_color = (0, 220, 255) if hold_ratio < 1.0 else (0, 255, 100)
        cv2.putText(frame, "HOLD:", (bar_x, bar_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x + 75, bar_y + 16), (bar_x + 185, bar_y + 28), (45, 45, 45), -1)
        cv2.rectangle(frame, (bar_x + 75, bar_y + 16), (bar_x + 75 + hold_w, bar_y + 28), hold_color, -1)

    # Voice status badge
    tts_text = "VOICE: [ON]" if tts_enabled else "VOICE: [OFF]"
    tts_color = (0, 220, 120) if tts_enabled else (120, 120, 120)
    cv2.putText(frame, tts_text, (w - 125, h - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.48, tts_color, 1, cv2.LINE_AA)

    # Row 2: Sentence Bar
    cv2.line(frame, (10, h - 78), (w - 10, h - 78), (45, 45, 45), 1)
    cv2.putText(frame, "SENTENCE:", (14, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

    sent_display = sentence if sentence != "..." else "Add words by signing..."
    sent_color = (255, 255, 255) if sentence != "..." else (110, 110, 110)
    cv2.putText(frame, sent_display, (100, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sent_color, 2, cv2.LINE_AA)

    # Row 3: Command Controls Footer
    footer = "ENTER=Speak Sentence   BACKSPACE=Delete Word   C=Clear   M=Mic STT   T=Voice   Q=Quit"
    cv2.putText(frame, footer, (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)


def main():
    print("[JARVIS] Initializing camera, hand tracking, and speech...")
    cam = CameraStream(camera_index=0, width=640, height=480)
    detector = HandDetector(max_hands=2, min_detection_conf=0.7)

    if not cam.start():
        print("[JARVIS] Could not start camera. Exiting.")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"[JARVIS] Model not found at {MODEL_PATH}")
        print("[JARVIS] Please run python src/ml/train_models.py first.")
        cam.release()
        sys.exit(1)

    classifier = SignClassifier(MODEL_PATH, smooth_window=5, confidence_threshold=0.65)
    speech = SpeechEngine(rate=2, volume=100)
    listener = SpeechListener()
    sentence_builder = SentenceBuilder(max_words=12)

    tts_enabled = True
    REQUIRED_STABLE_FRAMES = 9
    current_candidate = None
    hold_count = 0
    last_spoken_sign = None
    spoken_flash_timer = 0.0

    print("[JARVIS] Two-Way Translator is online!")
    print("         Sign to form sentences. Press M to listen to speech.")

    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        fps = cam.compute_fps()

        # Hand detection & landmarks
        detector.process_frame(frame)
        detector.draw_landmarks(frame)

        n_hands = detector.num_hands_detected()
        hands_lbl = detector.get_handedness_list()
        features_126 = detector.extract_features_both()

        # ML Prediction with temporal smoothing
        if n_hands > 0:
            pred_sign, confidence = classifier.predict_smooth(features_126)

            if pred_sign != "..." and confidence >= 0.70:
                if pred_sign == current_candidate:
                    hold_count += 1
                else:
                    current_candidate = pred_sign
                    hold_count = 1

                # Confirmed after holding stable
                if hold_count >= REQUIRED_STABLE_FRAMES:
                    if pred_sign != last_spoken_sign or (time.time() - spoken_flash_timer) > 2.5:
                        if tts_enabled:
                            speech.speak(pred_sign)
                        # Auto-append to sentence builder
                        sentence_builder.add_word(pred_sign)
                        last_spoken_sign = pred_sign
                        spoken_flash_timer = time.time()
            else:
                current_candidate = None
                hold_count = 0
        else:
            classifier.reset_history()
            current_candidate = None
            hold_count = 0
            last_spoken_sign = None
            pred_sign, confidence = "...", 0.0

        hold_ratio = min(1.0, hold_count / REQUIRED_STABLE_FRAMES) if (current_candidate and hold_count > 0) else 0.0

        # Draw HUD
        draw_hud(
            frame=frame,
            sign=pred_sign,
            conf=confidence,
            n_hands=n_hands,
            hands_lbl=hands_lbl,
            fps=fps,
            tts_enabled=tts_enabled,
            model_name=classifier.model_name,
            hold_ratio=hold_ratio,
            sentence=sentence_builder.get_sentence(),
            stt_status=listener.status_msg,
            stt_text=listener.last_text,
            is_listening=listener.is_listening,
        )

        cv2.imshow("JARVIS — Universal Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("t"), ord("T")):
            tts_enabled = not tts_enabled
            state = "ON" if tts_enabled else "OFF"
            print(f"[JARVIS] Voice Speech: {state}")
        elif key in (ord("m"), ord("M")):
            # Start Speech-to-Text microphone recording (3.5s)
            print("[JARVIS] Listening to microphone...")
            listener.listen_async(duration_sec=3.5)
        elif key == 13:  # ENTER key -> Speak entire sentence
            sent = sentence_builder.get_sentence()
            if sent != "...":
                print(f"[JARVIS] Speaking Full Sentence: '{sent}'")
                speech.speak(sent, force=True)
        elif key in (8, 127):  # BACKSPACE key -> Remove last word
            sentence_builder.remove_last()
            print("[JARVIS] Removed last word from sentence")
        elif key in (ord("c"), ord("C")):  # C key -> Clear sentence
            sentence_builder.clear()
            listener.clear()
            print("[JARVIS] Cleared sentence & speech text")
        elif key in (ord("s"), ord("S")):
            if pred_sign != "...":
                speech.speak(pred_sign, force=True)

    # Cleanup
    print("\n[JARVIS] Shutting down...")
    speech.stop()
    detector.close()
    cam.release()
    cv2.destroyAllWindows()
    print("[JARVIS] Goodbye.")


if __name__ == "__main__":
    main()
