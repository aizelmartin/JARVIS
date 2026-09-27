"""
main.py — JARVIS Entry Point
=============================
Run this file to launch JARVIS.

Current Phase : 2 — Dual-Hand Landmark Detection
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
    detector = HandDetector(max_hands=2, min_detection_conf=0.7)

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

        n_hands  = detector.num_hands_detected()
        hands_lbl = detector.get_handedness_list()   # e.g. ['Left', 'Right']

        # Single-hand features (63) for A–Z classifier
        feat_single = detector.extract_features_single()
        # Both-hand features (126) for two-hand sign classifier
        feat_both   = detector.extract_features_both()

        # ── Overlays ─────────────────────────────────────────────────── #
        CameraStream.draw_fps(frame, fps)

        if n_hands == 0:
            status_text  = "No hands detected — show your hand(s)"
            status_color = (0, 100, 255)      # orange-red
        elif n_hands == 1:
            status_text  = f"1 hand detected: {hands_lbl[0]}  |  63-feature vector ready"
            status_color = (0, 220, 80)       # green
        else:
            status_text  = f"2 hands: {hands_lbl}  |  126-feature vector ready"
            status_color = (255, 180, 0)      # gold — two-hand mode

        cv2.putText(frame, status_text, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

        # Show a sample of the combined feature vector (first 6 of 126)
        if n_hands > 0:
            preview = "  ".join([f"{v:+.2f}" for v in feat_both[:6]])
            cv2.putText(frame, f"features[0..5]: {preview}", (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # Mode badge
        mode = "TWO-HAND MODE" if n_hands == 2 else ("SINGLE-HAND" if n_hands == 1 else "IDLE")
        badge_color = (255, 180, 0) if n_hands == 2 else ((0, 220, 80) if n_hands == 1 else (80, 80, 80))
        cv2.putText(frame, mode, (frame.shape[1] - 190, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, badge_color, 2, cv2.LINE_AA)

        cv2.putText(frame, "JARVIS — Phase 2: Dual-Hand Tracking", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

        cv2.imshow("JARVIS", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    detector.close()
    cam.release()
    print("[JARVIS] Goodbye.")


if __name__ == "__main__":
    main()
