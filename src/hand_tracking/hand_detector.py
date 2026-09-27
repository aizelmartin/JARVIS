"""
hand_detector.py
----------------
Detects hands in a BGR frame using MediaPipe, draws landmarks,
and extracts a normalized 63-dimensional feature vector suitable
for training classical ML classifiers.

Normalization strategy (important for academic explanation):
  1. Translation invariance  — subtract wrist (landmark 0) from all 21 points
  2. Scale invariance        — divide by the maximum absolute value across all
                               coordinates so the vector lives in [-1, 1]

This means predictions are NOT affected by:
  - Where the hand appears in the frame
  - How close the hand is to the camera
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional


class HandDetector:
    """
    Wraps MediaPipe Hands for single-hand detection and feature extraction.

    Args:
        max_hands (int):          Maximum number of hands to detect (keep 1 for ASL
                                  static gesture recognition).
        min_detection_conf (float): Minimum confidence to consider a detection valid.
        min_tracking_conf  (float): Minimum confidence to keep tracking an existing hand.
    """

    # MediaPipe provides 21 landmarks, each with (x, y, z) → 63 features total
    NUM_LANDMARKS = 21
    NUM_FEATURES  = NUM_LANDMARKS * 3   # 63

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_conf: float = 0.7,
        min_tracking_conf: float = 0.5,
    ):
        self._mp_hands    = mp.solutions.hands
        self._mp_drawing  = mp.solutions.drawing_utils
        self._mp_styles   = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,          # False = optimised for video stream
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )

        # Cached results from the last frame
        self._results = None

    # ------------------------------------------------------------------ #
    #  Core processing
    # ------------------------------------------------------------------ #

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Runs MediaPipe on a BGR frame.

        Args:
            frame_bgr: OpenCV frame (H x W x 3, dtype uint8).

        Returns:
            The same frame object (results stored internally).
        """
        # MediaPipe requires RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Mark the frame as not writeable to improve performance
        rgb.flags.writeable = False
        self._results = self._hands.process(rgb)
        rgb.flags.writeable = True

        return frame_bgr

    # ------------------------------------------------------------------ #
    #  Drawing
    # ------------------------------------------------------------------ #

    def draw_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Draws hand skeleton (connections + landmark dots) onto the frame.

        Args:
            frame_bgr: BGR frame (modified in-place).

        Returns:
            The annotated frame.
        """
        if self._results and self._results.multi_hand_landmarks:
            for hand_landmarks in self._results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )
        return frame_bgr

    # ------------------------------------------------------------------ #
    #  Feature extraction
    # ------------------------------------------------------------------ #

    def extract_features(self) -> Optional[np.ndarray]:
        """
        Returns a normalized 63-dimensional feature vector from the first
        detected hand, or None if no hand is visible.

        Normalization:
          Step 1 — Translation: Subtract the wrist landmark (index 0) from
                   every landmark so the wrist is at the origin (0, 0, 0).
          Step 2 — Scale: Divide every value by the maximum absolute coordinate
                   so all values fall inside [-1.0, 1.0].

        Returns:
            np.ndarray of shape (63,) and dtype float32, or None.
        """
        if not self._results or not self._results.multi_hand_landmarks:
            return None

        # Use only the first detected hand
        hand_landmarks = self._results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark   # list of 21 NormalizedLandmark objects

        # --- Step 1: collect raw (x, y, z) ---
        raw = np.array(
            [[p.x, p.y, p.z] for p in lm],
            dtype=np.float32
        )  # shape: (21, 3)

        # --- Step 2: translation invariance — make wrist the origin ---
        wrist = raw[0].copy()
        raw   = raw - wrist             # shape: (21, 3)

        # --- Step 3: scale invariance ---
        max_val = np.max(np.abs(raw))
        if max_val > 0:
            raw = raw / max_val         # now in [-1, 1]

        # --- Flatten to (63,) ---
        features = raw.flatten()
        return features

    # ------------------------------------------------------------------ #
    #  Detection status helpers
    # ------------------------------------------------------------------ #

    def hand_detected(self) -> bool:
        """Returns True if at least one hand was found in the last frame."""
        return bool(
            self._results
            and self._results.multi_hand_landmarks
        )

    def get_handedness(self) -> Optional[str]:
        """
        Returns 'Left' or 'Right' for the first detected hand.

        Note: Because we mirror the frame horizontally in CameraStream,
        Left and Right appear swapped relative to MediaPipe's native labels —
        we intentionally flip them back here.
        """
        if not self._results or not self._results.multi_handedness:
            return None
        label = self._results.multi_handedness[0].classification[0].label
        # Mirror correction
        return "Left" if label == "Right" else "Right"

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #

    def close(self):
        """Releases MediaPipe resources."""
        self._hands.close()
