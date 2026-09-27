"""
main.py — JARVIS Universal Sign Language Translator
===================================================
Real-time webcam pipeline:
  1. Camera Stream: 640x480 high-FPS video feed
  2. Hand Tracking: Dual-hand MediaPipe tracking & landmark normalization (126 features)
  3. ML Classifier: Real-time inference with temporal smoothing majority voting
  4. Real-Time Speech: Hold-to-Speak (Dwell Time) gesture stabilization with zero audio lag

Controls:
  Q — Quit
  T — Toggle Speech (TTS) ON / OFF
  S — Force speak current sign immediately
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


def draw_hud(
    frame,
    sign: str,
    conf: float,
    n_hands: int,
    hands_lbl: list,
    fps: float,
    tts_enabled: bool,
    model_name: str,
    hold_ratio: float = 0.0,
    just_spoken: bool = False,
):
    """Renders an intuitive, responsive HUD over the camera frame."""
    h, w = frame.shape[:2]

    # Top status banner
    cv2.rectangle(frame, (0, 0), (w, 42), (20, 20, 20), -1)
    cv2.line(frame, (0, 42), (w, 42), (70, 70, 70), 1)

    # Title & FPS
    cv2.putText(frame, "JARVIS", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, f"|  FPS: {fps:.1f}", (110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"|  Model: {model_name}", (210, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 200, 240), 1, cv2.LINE_AA)

    # Hand mode badge (Top Right)
    mode_text = f"{n_hands} HAND(S)" if n_hands > 0 else "IDLE"
    mode_color = (0, 220, 120) if n_hands == 2 else ((255, 180, 0) if n_hands == 1 else (100, 100, 100))
    cv2.putText(frame, mode_text, (w - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2, cv2.LINE_AA)

    # Bottom Prediction Card
    card_h = 105
    cv2.rectangle(frame, (0, h - card_h), (w, h), (18, 18, 18), -1)
    cv2.line(frame, (0, h - card_h), (w, h - card_h), (60, 60, 60), 1)

    # Sign label display
    has_sign = (sign != "..." and conf >= 0.60)
    display_sign = sign.upper() if has_sign else "..."
    sign_color = (0, 240, 140) if has_sign else (120, 120, 120)

    cv2.putText(frame, "PREDICTED SIGN:", (14, h - 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, display_sign, (14, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, sign_color, 3, cv2.LINE_AA)

    # Speech Spoken Indicator
    if just_spoken:
        cv2.rectangle(frame, (14, h - 22), (160, h - 5), (0, 180, 90), -1)
        cv2.putText(frame, "🔊 SPOKEN ALOUD", (20, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # Hold / Speak Dwell Meter
    if has_sign:
        # Confidence meter
        bar_x, bar_y = 280, h - 68
        bar_w, bar_max = int(160 * conf), 160
        cv2.putText(frame, f"CONFIDENCE: {conf*100:.1f}%", (bar_x, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max, bar_y + 10), (45, 45, 45), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (0, 210, 120), -1)

        # Hold stability meter (fills to trigger speech)
        hold_y = h - 28
        hold_w = int(160 * hold_ratio)
        meter_label = "HOLD STEADY TO SPEAK:" if hold_ratio < 1.0 else "CONFIRMED & SPOKEN:"
        meter_color = (0, 200, 255) if hold_ratio < 1.0 else (0, 240, 100)
        cv2.putText(frame, meter_label, (bar_x, hold_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x, hold_y), (bar_x + bar_max, hold_y + 10), (45, 45, 45), -1)
        cv2.rectangle(frame, (bar_x, hold_y), (bar_x + hold_w, hold_y + 10), meter_color, -1)

    # TTS Status badge (Bottom Right)
    tts_text = "VOICE: [ON]" if tts_enabled else "VOICE: [OFF]"
    tts_color = (0, 220, 120) if tts_enabled else (120, 120, 120)
    cv2.putText(frame, tts_text, (w - 130, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tts_color, 2, cv2.LINE_AA)

    # Instructions footer
    footer = "Q=Quit   T=Toggle Voice   S=Speak Now"
    cv2.putText(frame, footer, (w - 270, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)


def main():
    print("[JARVIS] Initializing camera and hand tracking...")
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

    # Classifier with small temporal window for quick reaction
    classifier = SignClassifier(MODEL_PATH, smooth_window=5, confidence_threshold=0.65)
    speech = SpeechEngine(rate=2, volume=100)
    tts_enabled = True

    # ── Real-Time Gesture Hold-to-Speak (Dwell) State ────────────────── #
    # At ~30 FPS, 9 consecutive frames = ~0.30 seconds hold to confirm
    REQUIRED_STABLE_FRAMES = 9
    current_candidate = None
    hold_count = 0
    last_spoken_sign = None
    spoken_flash_timer = 0.0

    print("[JARVIS] Real-time translator initialized.")
    print("         Hold a gesture steady for 0.3s to speak it.")
    print("         Keys: Q=Quit, T=Toggle Voice, S=Speak current sign.")

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

            # Hold-to-Speak stabilization logic
            if pred_sign != "..." and confidence >= 0.70:
                if pred_sign == current_candidate:
                    hold_count += 1
                else:
                    current_candidate = pred_sign
                    hold_count = 1

                # Confirmed after holding stable
                if hold_count >= REQUIRED_STABLE_FRAMES:
                    # Speak if it's a new sign OR if held continuously for > 2.5s
                    if pred_sign != last_spoken_sign or (time.time() - spoken_flash_timer) > 2.5:
                        if tts_enabled:
                            speech.speak(pred_sign)
                        last_spoken_sign = pred_sign
                        spoken_flash_timer = time.time()
            else:
                current_candidate = None
                hold_count = 0
        else:
            classifier.reset_history()
            current_candidate = None
            hold_count = 0
            last_spoken_sign = None  # reset so returning to the sign will speak again
            pred_sign, confidence = "...", 0.0

        hold_ratio = min(1.0, hold_count / REQUIRED_STABLE_FRAMES) if (current_candidate and hold_count > 0) else 0.0
        just_spoken = (time.time() - spoken_flash_timer) < 1.5

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
            hold_ratio=hold_ratio,
            just_spoken=just_spoken,
        )

        cv2.imshow("JARVIS — Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("t"), ord("T")):
            tts_enabled = not tts_enabled
            state = "ON" if tts_enabled else "OFF"
            print(f"[JARVIS] Voice Speech: {state}")
        elif key in (ord("s"), ord("S")):
            if pred_sign != "...":
                speech.speak(pred_sign, force=True)
                spoken_flash_timer = time.time()

    # Cleanup
    print("\n[JARVIS] Shutting down...")
    speech.stop()
    detector.close()
    cam.release()
    cv2.destroyAllWindows()
    print("[JARVIS] Goodbye.")


if __name__ == "__main__":
    main()
