"""
merge_datasets.py
-----------------
Merges landmarks_custom.csv (5 custom words, 126 features = dual-hand)
with landmarks_alphabet.csv (ASL A-Z letters, 63 features = single-hand)
into a single combined training CSV with a unified 126-feature schema.

Strategy:
  - Alphabet rows (63 features) are zero-padded to 126 features.
    Zeros represent "no second hand detected", which is consistent with
    how the live system generates features for single-hand signs.

Drops non-sign classes: 'nothing', 'del', 'space'
Also drops severely under-represented classes if below MIN_SAMPLES.

Output:
    data/processed/landmarks_combined.csv

Usage:
    python src/dataset/merge_datasets.py
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CUSTOM_CSV   = os.path.join(PROJECT_ROOT, "data", "processed", "landmarks_custom.csv")
ALPHABET_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "landmarks_alphabet.csv")
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "data", "processed", "landmarks_combined.csv")

# Drop these dataset artifacts — not real ASL gestures
DROP_CLASSES = {"nothing", "del", "space"}

# Drop any class with fewer than this many samples
MIN_SAMPLES = 20

# Unified feature count (matches live dual-hand pipeline)
NUM_FEATURES = 126
FEATURE_COLS = [f"f{i}" for i in range(NUM_FEATURES)]


def main():
    print("[Merge] Loading datasets...")

    if not os.path.exists(CUSTOM_CSV):
        print(f"[Merge] ERROR: Custom dataset not found: {CUSTOM_CSV}")
        sys.exit(1)

    if not os.path.exists(ALPHABET_CSV):
        print(f"[Merge] ERROR: Alphabet dataset not found: {ALPHABET_CSV}")
        sys.exit(1)

    df_custom   = pd.read_csv(CUSTOM_CSV)
    df_alphabet = pd.read_csv(ALPHABET_CSV)

    print(f"[Merge] Custom dataset   : {len(df_custom):>5} rows, {len(df_custom.columns)-1} features")
    print(f"[Merge] Alphabet dataset : {len(df_alphabet):>5} rows, {len(df_alphabet.columns)-1} features")

    # ── Normalize custom dataset to 126 features ───────────────────────── #
    custom_feat_cols = [c for c in df_custom.columns if c != "label"]
    if len(custom_feat_cols) > NUM_FEATURES:
        # Trim if somehow wider
        custom_feat_cols = custom_feat_cols[:NUM_FEATURES]
    df_custom_norm = pd.DataFrame(
        df_custom[custom_feat_cols].values,
        columns=[f"f{i}" for i in range(len(custom_feat_cols))]
    )
    # Pad if needed (shouldn't be necessary for custom)
    for i in range(len(custom_feat_cols), NUM_FEATURES):
        df_custom_norm[f"f{i}"] = 0.0
    df_custom_norm = df_custom_norm[FEATURE_COLS]
    df_custom_norm["label"] = df_custom["label"].values
    print(f"[Merge] Custom normalized  -> {df_custom_norm.shape[1]-1} features")

    # ── Normalize alphabet dataset: 63 -> 126 (zero-pad 2nd hand) ─────── #
    alphabet_feat_cols = [c for c in df_alphabet.columns if c != "label"]
    df_alphabet_norm = pd.DataFrame(
        df_alphabet[alphabet_feat_cols].values,
        columns=[f"f{i}" for i in range(len(alphabet_feat_cols))]
    )
    # Zero-pad missing features (f63 through f125)
    for i in range(len(alphabet_feat_cols), NUM_FEATURES):
        df_alphabet_norm[f"f{i}"] = 0.0
    df_alphabet_norm = df_alphabet_norm[FEATURE_COLS]
    df_alphabet_norm["label"] = df_alphabet["label"].values
    print(f"[Merge] Alphabet normalized -> {df_alphabet_norm.shape[1]-1} features (zero-padded for 2nd hand)")

    # ── Drop unwanted alphabet classes ─────────────────────────────────── #
    before = len(df_alphabet_norm)
    df_alphabet_norm = df_alphabet_norm[~df_alphabet_norm["label"].isin(DROP_CLASSES)]
    dropped_rows = before - len(df_alphabet_norm)
    print(f"[Merge] Dropped {dropped_rows} rows for classes {DROP_CLASSES}")

    # ── Concatenate ────────────────────────────────────────────────────── #
    # [JARVIS] User requested to test custom words only.
    # df_combined = pd.concat([df_custom_norm, df_alphabet_norm], ignore_index=True)
    df_combined = df_custom_norm.copy()

    # Sanity check — no NaNs should exist
    nan_count = df_combined.isnull().sum().sum()
    if nan_count > 0:
        print(f"[Merge] WARNING: {nan_count} NaN values found — filling with 0")
        df_combined.fillna(0.0, inplace=True)
    else:
        print(f"[Merge] NaN check passed - no missing values")

    # ── Drop under-represented classes ─────────────────────────────────── #
    counts = df_combined["label"].value_counts()
    low_classes = counts[counts < MIN_SAMPLES].index.tolist()
    if low_classes:
        print(f"[Merge] Dropping under-represented classes (< {MIN_SAMPLES} samples): {low_classes}")
        df_combined = df_combined[~df_combined["label"].isin(low_classes)]

    # ── Shuffle ────────────────────────────────────────────────────────── #
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n[Merge] Combined dataset : {len(df_combined)} rows x {NUM_FEATURES} features")
    print("[Merge] Class distribution:")
    dist = df_combined["label"].value_counts().sort_index()
    for label, count in dist.items():
        bar = "#" * (count // 10)
        print(f"         {label:12s} : {count:4d}  {bar}")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[Merge] Saved -> {OUTPUT_CSV}")
    print(f"[Merge] Total classes : {df_combined['label'].nunique()}")
    print("[Merge] Done.")


if __name__ == "__main__":
    main()
