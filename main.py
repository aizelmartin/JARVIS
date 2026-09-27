"""
main.py — JARVIS Entry Point
=============================
Run this file to launch JARVIS.

Current Phase : 1 — Camera Stream Test
  Press  Q   to quit the window.
"""

import cv2
import sys
import os

# Make sure 'src/' is importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))

from src.camera.camera_stream import CameraStream


def main():
    cam = CameraStream(camera_index=0, width=640, height=480)

    if not cam.start():
        print("[JARVIS] Could not start camera. Exiting.")
        sys.exit(1)

    print("[JARVIS] Press Q inside the window to quit.")

    while True:
        frame = cam.read_frame()
        if frame is None:
            print("[JARVIS] No frame received. Exiting.")
            break

        fps = cam.compute_fps()

        # Draw FPS on the frame
        CameraStream.draw_fps(frame, fps)

        # Draw a welcome banner
        cv2.putText(
            frame,
            "JARVIS - Sign Language Translator",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 200, 0),   # Gold / Amber
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Phase 1: Camera OK  |  Press Q to quit",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),  # Light grey
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("JARVIS", frame)

        # Quit on 'q' or 'Q'
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cam.release()
    print("[JARVIS] Goodbye.")


if __name__ == "__main__":
    main()
