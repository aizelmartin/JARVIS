"""
main.py — JARVIS Entry Point
=============================
Run this file to launch JARVIS.

Current Phase : 2 — Hand Landmark Detection & Normalization
  Press  Q   to quit.
"""

import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.camera.camera_stream import CameraStream
from src.hand_tracking.hand_detector import HandDetector


def main():
    cam      = CameraStream(camera_index=0, width=640, height=480)
    detector = HandDetector(max_hands=1, min_detection_conf=0.7)

    if not cam.start():
        print("[JARVIS] Could not start camera. Exiting.")
        sys.exit(1)

    print("[JARVIS] Press Q inside the window to quit.")

    while True:
        frame = cam.read_frame()
        if frame is None:
            break

        fps = cam.compute_fps()

        # ── MediaPipe processing ──────────────────────────────────────── #
        detector.process_frame(frame)
        detector.draw_landmarks(frame)
        features = detector.extract_features()   # shape (63,) or None

        # ── Overlays ─────────────────────────────────────────────────── #
        CameraStream.draw_fps(frame, fps)

        # Hand status banner
        if detector.hand_detected():
            side = detector.get_handedness() or "?"
            label = f"Hand detected ({side})  |  Features: {len(features)} dims"
            color = (0, 255, 120)       # Green
        else:
            label = "No hand detected — show your hand"
            color = (0, 100, 255)       # Orange-red

        cv2.putText(frame, label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Show first 6 normalized feature values for visual confirmation
        if features is not None:
            feat_text = "  ".join([f"{v:+.2f}" for v in features[:6]])
            cv2.putText(frame, f"f[0..5]: {feat_text}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.putText(frame, "JARVIS - Phase 2: Hand Tracking", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)

        cv2.imshow("JARVIS", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    detector.close()
    cam.release()
    print("[JARVIS] Goodbye.")


if __name__ == "__main__":
    main()
