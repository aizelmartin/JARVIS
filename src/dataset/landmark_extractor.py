"""
landmark_extractor.py
----------------------
Batch-processes a folder of ASL Alphabet images through MediaPipe and
exports a clean landmark CSV file ready for ML training.

Input folder structure expected (standard Kaggle ASL Alphabet layout):
    data/raw/asl_alphabet_train/
        A/
            A1.jpg
            A2.jpg
            ...
        B/
            B1.jpg
            ...
        Z/
            ...
        del/
            ...
        nothing/
            ...
        space/
            ...

Output:
    data/processed/landmarks_alphabet.csv
    Columns: f0, f1, ..., f62, label
    Each row = one image where MediaPipe detected a hand.
    Rows where no hand was detected are silently skipped.

Usage:
    python -m src.dataset.landmark_extractor
    (or run directly: python src/dataset/landmark_extractor.py)
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import sys
import time

# ── Paths ─────────────────────────────────────────────────────────────── #
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw", "asl_alphabet_train")
OUTPUT_CSV    = os.path.join(PROJECT_ROOT, "data", "processed", "landmarks_alphabet.csv")

# ── MediaPipe (static image mode = best accuracy for still images) ─────── #
_mp_hands = mp.solutions.hands
_HANDS    = _mp_hands.Hands(
    static_image_mode=True,        # process each image independently
    max_num_hands=1,
    min_detection_confidence=0.5,  # lower threshold for dataset images
)

NUM_FEATURES = 63    # 21 landmarks × 3 axes


def normalize_hand(landmarks) -> np.ndarray:
    """Translation + scale normalization → 63-dim float32 vector."""
    raw = np.array(
        [[p.x, p.y, p.z] for p in landmarks.landmark],
        dtype=np.float32,
    )
    raw -= raw[0]                      # wrist to origin
    max_val = np.max(np.abs(raw))
    if max_val > 0:
        raw /= max_val
    return raw.flatten()


def extract_from_image(image_path: str) -> np.ndarray | None:
    """
    Run MediaPipe on a single image file.

    Returns:
        63-dim feature vector or None if no hand detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = _HANDS.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    return normalize_hand(results.multi_hand_landmarks[0])


def extract_dataset(
    raw_dir: str = RAW_DIR,
    output_csv: str = OUTPUT_CSV,
    max_per_class: int | None = None,
) -> pd.DataFrame:
    """
    Walk every class folder under raw_dir, extract landmarks, and save CSV.

    Args:
        raw_dir       : Root folder containing one sub-folder per class.
        output_csv    : Path to save the resulting CSV.
        max_per_class : If set, limit samples per class (useful for quick tests).
                        Set None to use all images.

    Returns:
        The resulting DataFrame.
    """
    if not os.path.isdir(raw_dir):
        print(f"[Extractor] ERROR: Dataset folder not found: {raw_dir}")
        print("[Extractor] Please download the ASL Alphabet dataset first.")
        print("[Extractor] See README / Phase 3 instructions.")
        sys.exit(1)

    class_folders = sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ])

    if not class_folders:
        print(f"[Extractor] No class sub-folders found in {raw_dir}")
        sys.exit(1)

    print(f"[Extractor] Found {len(class_folders)} classes: {class_folders}")
    print(f"[Extractor] Output → {output_csv}")
    if max_per_class:
        print(f"[Extractor] Limiting to {max_per_class} images per class")
    print()

    rows         = []
    col_names    = [f"f{i}" for i in range(NUM_FEATURES)] + ["label"]
    total_found  = 0
    total_skipped = 0
    t_start      = time.time()

    for cls_idx, cls_name in enumerate(class_folders):
        cls_dir  = os.path.join(raw_dir, cls_name)
        images   = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if max_per_class:
            images = images[:max_per_class]

        found   = 0
        skipped = 0

        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)
            features = extract_from_image(img_path)

            if features is None:
                skipped += 1
                continue

            rows.append(list(features) + [cls_name])
            found += 1

        total_found   += found
        total_skipped += skipped

        elapsed = time.time() - t_start
        pct     = (cls_idx + 1) / len(class_folders) * 100
        print(
            f"  [{cls_idx+1:2d}/{len(class_folders)}] {cls_name:8s} "
            f"✓ {found:5d}  ✗ {skipped:4d}  |  "
            f"total so far: {total_found:6d}  |  "
            f"{pct:5.1f}%  {elapsed:.0f}s elapsed"
        )

    print()
    print(f"[Extractor] Done — {total_found} samples extracted, {total_skipped} skipped.")
    print(f"[Extractor] Saving → {output_csv}")

    df = pd.DataFrame(rows, columns=col_names)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"[Extractor] Saved {len(df)} rows × {len(df.columns)} cols")
    print(f"[Extractor] Class distribution:")
    print(df["label"].value_counts().to_string())
    return df


# ── Entry point ───────────────────────────────────────────────────────── #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from ASL images")
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Limit samples per class (e.g. 500 for a quick test run)",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=RAW_DIR,
        help="Path to the dataset root folder",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_CSV,
        help="Output CSV path",
    )
    args = parser.parse_args()

    extract_dataset(
        raw_dir=args.raw_dir,
        output_csv=args.output,
        max_per_class=args.max_per_class,
    )
