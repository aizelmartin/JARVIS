"""
data_collector.py
------------------
Live webcam tool to record custom sign samples for the two-hand classifier.

Use this to collect data for custom signs like: Hello, Thank You, Help, etc.

Controls:
    S     — Start recording samples for the current sign
    SPACE — Save one sample (or it auto-saves when recording is active)
    N     — Move to next sign
    Q     — Quit

Output:
    data/processed/landmarks_custom.csv
    Columns: f0 .. f125, label    (126 features for 2-hand = left(63) + right(63))

Usage:
    python src/dataset/data_collector.py

Optionally customize the sign list below.
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.camera.camera_stream import CameraStream
from src.hand_tracking.hand_detector import HandDetector

OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "landmarks_custom.csv")

# ── Configure your custom signs here ─────────────────────────────────── #
CUSTOM_SIGNS     = ["hello", "thankyou", "help", "yes", "no"]
SAMPLES_PER_SIGN = 200          # samples to collect per sign
AUTO_RECORD_FPS  = 10           # auto-capture rate when recording (frames/sec)


def overlay_bar(frame, filled: int, total: int, y: int = 110):
    """Draw a progress bar showing collection progress."""
    bar_w   = 300
    bar_h   = 18
    x_start = 10
    pct     = filled / total if total > 0 else 0
    cv2.rectangle(frame, (x_start, y), (x_start + bar_w, y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x_start, y), (x_start + int(bar_w * pct), y + bar_h), (0, 200, 100), -1)
    cv2.putText(frame, f"{filled}/{total}", (x_start + bar_w + 8, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)


def run_collector():
    cam      = CameraStream(camera_index=0, width=640, height=480)
    detector = HandDetector(max_hands=2, min_detection_conf=0.6)

    if not cam.start():
        print("[Collector] Could not open camera.")
        sys.exit(1)

    col_names = [f"f{i}" for i in range(126)] + ["label"]
    rows      = []

    # Load existing data so we can append to it
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        rows     = existing.values.tolist()
        print(f"[Collector] Loaded {len(rows)} existing samples from {OUTPUT_CSV}")

    sign_idx    = 0
    recording   = False
    sign_count  = 0
    last_capture = 0.0
    capture_interval = 1.0 / AUTO_RECORD_FPS

    print(f"[Collector] Signs to collect: {CUSTOM_SIGNS}")
    print(f"[Collector] {SAMPLES_PER_SIGN} samples per sign")
    print()

    while sign_idx < len(CUSTOM_SIGNS):
        current_sign = CUSTOM_SIGNS[sign_idx]

        # Count how many we already have for this sign
        existing_for_sign = sum(1 for r in rows if r[-1] == current_sign)
        sign_count = existing_for_sign

        frame = cam.read_frame()
        if frame is None:
            break

        fps = cam.compute_fps()
        detector.process_frame(frame)
        detector.draw_landmarks(frame)

        features_126 = detector.extract_features_both()
        n_hands      = detector.num_hands_detected()
        hands_lbl    = detector.get_handedness_list()

        # ── Auto-capture when recording ───────────────────────────── #
        now = time.time()
        if recording and n_hands > 0 and (now - last_capture) >= capture_interval:
            if sign_count < SAMPLES_PER_SIGN:
                rows.append(list(features_126) + [current_sign])
                sign_count += 1
                last_capture = now

            if sign_count >= SAMPLES_PER_SIGN:
                recording = False
                print(f"[Collector] ✅ '{current_sign}' complete — {sign_count} samples")
                # Auto-advance to next sign after a brief pause
                time.sleep(1.0)
                sign_idx += 1
                if sign_idx >= len(CUSTOM_SIGNS):
                    break
                continue

        # ── Overlays ─────────────────────────────────────────────── #
        CameraStream.draw_fps(frame, fps)

        # Sign label
        cv2.putText(frame, f"Sign: '{current_sign}'", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2, cv2.LINE_AA)

        # Progress bar
        overlay_bar(frame, sign_count, SAMPLES_PER_SIGN)

        # Hand status
        hand_txt   = f"{n_hands} hand(s): {hands_lbl}" if n_hands > 0 else "No hands detected"
        hand_color = (0, 220, 80) if n_hands > 0 else (0, 100, 255)
        cv2.putText(frame, hand_txt, (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1, cv2.LINE_AA)

        # Recording status
        if recording:
            rec_txt   = "● RECORDING — hold the sign steadily"
            rec_color = (0, 0, 255)
        else:
            remaining = SAMPLES_PER_SIGN - sign_count
            rec_txt   = f"Press S to start recording ({remaining} samples needed)"
            rec_color = (200, 200, 200)
        cv2.putText(frame, rec_txt, (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 1, cv2.LINE_AA)

        # Instructions
        instrs = "S=Record   N=Skip sign   Q=Quit & Save"
        cv2.putText(frame, instrs, (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Sign progress list on the right
        for i, s in enumerate(CUSTOM_SIGNS):
            cnt   = sum(1 for r in rows if r[-1] == s)
            done  = cnt >= SAMPLES_PER_SIGN
            color = (0, 220, 80) if done else ((255, 200, 0) if i == sign_idx else (100, 100, 100))
            mark  = "✓" if done else ("►" if i == sign_idx else " ")
            cv2.putText(frame, f"{mark} {s} ({cnt}/{SAMPLES_PER_SIGN})",
                        (frame.shape[1] - 220, 45 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        cv2.imshow("JARVIS — Custom Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("s"), ord("S")):
            if n_hands > 0:
                recording = True
                last_capture = 0.0
                print(f"[Collector] Recording '{current_sign}'...")
            else:
                print("[Collector] Show your hands first before recording!")
        elif key in (ord("n"), ord("N")):
            recording = False
            print(f"[Collector] Skipping '{current_sign}' (saved {sign_count} so far)")
            sign_idx += 1

    # ── Save ─────────────────────────────────────────────────────────── #
    detector.close()
    cam.release()

    if rows:
        df = pd.DataFrame(rows, columns=col_names)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[Collector] Saved {len(df)} total samples → {OUTPUT_CSV}")
        print(df["label"].value_counts().to_string())
    else:
        print("[Collector] No samples recorded.")

    print("[Collector] Done.")


if __name__ == "__main__":
    run_collector()
