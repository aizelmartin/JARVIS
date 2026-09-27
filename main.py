"""
main.py — JARVIS Universal Sign Language Translator
===================================================
Real-time webcam pipeline:
  1. Camera Stream: 640x480 video feed
  2. Hand Tracking: Dual-hand MediaPipe tracking & landmark normalization (126 features)
  3. ML Classifier: Real-time inference with temporal smoothing majority voting
  4. Speech Engine: Non-blocking Text-to-Speech feedback

Controls:
  Q — Quit
  T — Toggle Speech (TTS) ON / OFF
  S — Speak current sign immediately
"""

import cv2
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.camera.camera_stream import CameraStream
from src.hand_tracking.hand_detector import HandDetector
from src.ml.classifier import SignClassifier
from src.speech.speech_engine import SpeechEngine

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "saved_models", "sign_classifier.pkl")


def draw_hud(frame, sign: str, conf: float, n_hands: int, hands_lbl: list, fps: float, tts_enabled: bool, model_name: str):
    """Renders a sleek HUD over the camera frame."""
    h, w = frame.shape[:2]

    # Top dark banner for telemetry
    cv2.rectangle(frame, (0, 0), (w, 44), (20, 20, 20), -1)
    cv2.line(frame, (0, 44), (w, 44), (70, 70, 70), 1)

    # Title & FPS
    cv2.putText(frame, "JARVIS", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, f"|  FPS: {fps:.1f}", (110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"|  Model: {model_name}", (210, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 200, 240), 1, cv2.LINE_AA)

    # Hand mode badge (Top Right)
    mode_text = f"{n_hands} HAND(S)" if n_hands > 0 else "IDLE"
    mode_color = (0, 220, 120) if n_hands == 2 else ((255, 180, 0) if n_hands == 1 else (100, 100, 100))
    cv2.putText(frame, mode_text, (w - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2, cv2.LINE_AA)

    # Bottom Prediction Card
    card_h = 100
    cv2.rectangle(frame, (0, h - card_h), (w, h), (18, 18, 18), -1)
    cv2.line(frame, (0, h - card_h), (w, h - card_h), (60, 60, 60), 1)

    # Sign label display
    has_sign = (sign != "..." and conf >= 0.60)
    display_sign = sign.upper() if has_sign else "..."
    sign_color = (0, 240, 140) if has_sign else (120, 120, 120)

    cv2.putText(frame, "PREDICTED SIGN:", (14, h - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, display_sign, (14, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.25, sign_color, 3, cv2.LINE_AA)

    # Confidence bar
    if has_sign:
        bar_x, bar_y = 300, h - 35
        bar_w, bar_max = int(180 * conf), 180
        cv2.putText(frame, f"CONFIDENCE: {conf*100:.1f}%", (bar_x, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max, bar_y + 14), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (0, 210, 120), -1)

    # TTS Status badge (Bottom Right)
    tts_text = "TTS: [ON]" if tts_enabled else "TTS: [OFF]"
    tts_color = (0, 220, 120) if tts_enabled else (120, 120, 120)
    cv2.putText(frame, tts_text, (w - 130, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tts_color, 2, cv2.LINE_AA)

    # Instructions footer
    footer = "Keys:  Q=Quit   T=Toggle TTS   S=Speak Sign"
    cv2.putText(frame, footer, (w - 290, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)


def main():
    print("[JARVIS] Initializing camera and hand tracking...")
    cam = CameraStream(camera_index=0, width=640, height=480)
    detector = HandDetector(max_hands=2, min_detection_conf=0.7)

    if not cam.start():
        print("[JARVIS] Could not start camera. Exiting.")
        sys.exit(1)

    # Load ML classifier
    if not os.path.exists(MODEL_PATH):
        print(f"[JARVIS] Model not found at {MODEL_PATH}")
        print("[JARVIS] Please run python src/ml/train_models.py first.")
        cam.release()
        sys.exit(1)

    classifier = SignClassifier(MODEL_PATH, smooth_window=7, confidence_threshold=0.65)
    speech = SpeechEngine(rate=150, volume=1.0, repeat_cooldown=3.0)
    tts_enabled = True

    print("[JARVIS] Running real-time translator.")
    print("         Press Q to quit, T to toggle voice, S to force speak.")

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
            if tts_enabled and pred_sign != "..." and confidence >= 0.70:
                speech.speak(pred_sign)
        else:
            classifier.reset_history()
            pred_sign, confidence = "...", 0.0

        # Draw UI HUD
        draw_hud(
            frame=frame,
            sign=pred_sign,
            conf=confidence,
            n_hands=n_hands,
            hands_lbl=hands_lbl,
            fps=fps,
            tts_enabled=tts_enabled,
            model_name=classifier.model_name,
        )

        cv2.imshow("JARVIS — Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("t"), ord("T")):
            tts_enabled = not tts_enabled
            state = "ON" if tts_enabled else "OFF"
            print(f"[JARVIS] Text-to-Speech: {state}")
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
