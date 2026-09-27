"""
hand_detector.py
----------------
Detects up to 2 hands in a BGR frame using MediaPipe, draws landmarks,
and extracts normalized feature vectors for ML classification.

Two operating modes:
  - Single-hand mode  → 63 features  (21 landmarks × 3 axes)
                        Used for ASL A–Z alphabet classifier
  - Two-hand mode     → 126 features (42 landmarks × 3 axes)
                        Used for custom static two-hand signs
                        (Hello, Thank You, Help, etc.)

Normalization strategy (translation + scale invariance):
  1. For EACH hand independently:
       a) Subtract its own wrist (landmark 0) — removes position in frame
       b) Divide by max abs value — removes hand-to-camera distance
  2. Concatenate left-hand vector then right-hand vector (alphabetical label order)
     If a hand is absent, its 63 slots are filled with zeros.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, Dict


class HandDetector:
    """
    Wraps MediaPipe Hands for multi-hand detection and feature extraction.

    Args:
        max_hands          : 1 for alphabet mode, 2 for full hybrid mode.
        min_detection_conf : Minimum confidence for a valid detection.
        min_tracking_conf  : Minimum confidence to keep tracking.
    """

    NUM_LANDMARKS        = 21
    SINGLE_HAND_FEATURES = NUM_LANDMARKS * 3    # 63
    TWO_HAND_FEATURES    = NUM_LANDMARKS * 3 * 2 # 126

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_conf: float = 0.7,
        min_tracking_conf: float = 0.5,
    ):
        self._mp_hands   = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles  = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self._max_hands = max_hands
        self._results   = None

    # ------------------------------------------------------------------ #
    #  Core processing
    # ------------------------------------------------------------------ #

    def process_frame(self, frame_bgr: np.ndarray):
        """Run MediaPipe on a BGR frame. Results stored internally."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self._results = self._hands.process(rgb)
        rgb.flags.writeable = True
        return frame_bgr

    # ------------------------------------------------------------------ #
    #  Drawing
    # ------------------------------------------------------------------ #

    def draw_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draws skeleton for ALL detected hands onto frame (in-place)."""
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
    #  Feature extraction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_hand(landmarks) -> np.ndarray:
        """
        Normalize a single hand's landmarks into a 63-dim float32 vector.

        Steps:
          1. Collect raw (x, y, z) for all 21 landmarks.
          2. Subtract wrist (landmark 0) → translation invariance.
          3. Divide by max absolute value → scale invariance.

        Returns:
            np.ndarray of shape (63,) in range [-1, 1].
        """
        raw = np.array(
            [[p.x, p.y, p.z] for p in landmarks.landmark],
            dtype=np.float32,
        )                                  # shape (21, 3)

        raw -= raw[0]                      # wrist to origin

        max_val = np.max(np.abs(raw))
        if max_val > 0:
            raw /= max_val                 # scale to [-1, 1]

        return raw.flatten()               # shape (63,)

    # ------------------------------------------------------------------ #
    #  Public feature extraction API
    # ------------------------------------------------------------------ #

    def extract_features_single(self) -> Optional[np.ndarray]:
        """
        Returns a 63-dim feature vector for the FIRST detected hand.
        Returns None if no hand is visible.
        Use this for the A–Z alphabet classifier.
        """
        if not self._results or not self._results.multi_hand_landmarks:
            return None
        return self._normalize_hand(self._results.multi_hand_landmarks[0])

    def extract_features_both(self) -> np.ndarray:
        """
        Returns a 126-dim feature vector representing BOTH hands.

        Layout: [left_hand_63_features | right_hand_63_features]

        If only one hand is detected, the absent hand's 63 slots are zeros.
        If no hand is detected, returns all zeros.

        Use this for the two-hand custom sign classifier.
        """
        left_vec  = np.zeros(self.SINGLE_HAND_FEATURES, dtype=np.float32)
        right_vec = np.zeros(self.SINGLE_HAND_FEATURES, dtype=np.float32)

        if self._results and self._results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(self._results.multi_hand_landmarks):
                # MediaPipe labels are relative to its camera model;
                # we flip because our frame is mirror-flipped.
                label = self._results.multi_handedness[i].classification[0].label
                # "Right" in MediaPipe = Left in mirrored view, and vice versa
                corrected = "Left" if label == "Right" else "Right"

                vec = self._normalize_hand(hand_landmarks)
                if corrected == "Left":
                    left_vec = vec
                else:
                    right_vec = vec

        return np.concatenate([left_vec, right_vec])  # shape (126,)

    # ------------------------------------------------------------------ #
    #  Status helpers
    # ------------------------------------------------------------------ #

    def hand_detected(self) -> bool:
        """True if at least one hand was found in the last frame."""
        return bool(
            self._results and self._results.multi_hand_landmarks
        )

    def num_hands_detected(self) -> int:
        """Returns 0, 1, or 2 — how many hands were found."""
        if not self._results or not self._results.multi_hand_landmarks:
            return 0
        return len(self._results.multi_hand_landmarks)

    def get_handedness_list(self) -> list:
        """
        Returns a list of corrected labels for each detected hand.
        e.g. ['Right'] or ['Left', 'Right']
        """
        if not self._results or not self._results.multi_handedness:
            return []
        labels = []
        for h in self._results.multi_handedness:
            raw = h.classification[0].label
            labels.append("Left" if raw == "Right" else "Right")
        return labels

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #

    def close(self):
        """Releases MediaPipe resources."""
        self._hands.close()
