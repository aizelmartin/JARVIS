"""
classifier.py
-------------
Loads the trained ML model payload and provides real-time inference
with confidence scoring and temporal window smoothing to eliminate flicker.
"""

import os
import joblib
import numpy as np
from collections import deque, Counter
from typing import Tuple, Optional


class SignClassifier:
    """
    Inference wrapper for the trained sign language model.

    Features:
      - Real-time prediction with probability estimation
      - Rolling prediction buffer for temporal smoothing (majority vote)
      - Minimum confidence gating
    """

    def __init__(
        self,
        model_path: str,
        smooth_window: int = 7,
        confidence_threshold: float = 0.65,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.model_name = payload.get("model_name", "Unknown")
        self.label_encoder = payload["label_encoder"]
        self.classes = payload["classes"]
        self.num_features = payload["num_features"]

        self.confidence_threshold = confidence_threshold
        self.smooth_window = smooth_window
        self.history = deque(maxlen=smooth_window)

        print(f"[SignClassifier] Loaded '{self.model_name}' ({len(self.classes)} classes)")

    def predict_single(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        """
        Predicts a single feature vector without temporal smoothing.
        Returns: (predicted_class_name, confidence)
        """
        if feature_vector is None or len(feature_vector) != self.num_features:
            return "No Hand", 0.0

        # Check if all zeros (e.g. no hands)
        if np.all(feature_vector == 0):
            return "No Hand", 0.0

        sample = feature_vector.reshape(1, -1)

        # Predict probability if supported
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(sample)[0]
            best_idx = np.argmax(probs)
            confidence = float(probs[best_idx])
            pred_class = self.label_encoder.inverse_transform([best_idx])[0]
        else:
            pred_idx = self.model.predict(sample)[0]
            pred_class = self.label_encoder.inverse_transform([pred_idx])[0]
            confidence = 1.0

        return str(pred_class), confidence

    def predict_smooth(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        """
        Predicts with rolling-window majority voting to stabilize live output.
        Returns: (smoothed_class_name, average_confidence)
        """
        pred_label, conf = self.predict_single(feature_vector)

        if pred_label == "No Hand" or conf < self.confidence_threshold:
            self.history.append(("...", 0.0))
            return "...", conf

        self.history.append((pred_label, conf))

        # Majority vote across recent window
        labels = [item[0] for item in self.history if item[0] != "..."]
        if not labels:
            return "...", 0.0

        counter = Counter(labels)
        winner_label, count = counter.most_common(1)[0]

        # Calculate average confidence for the winning label in the window
        winner_confs = [item[1] for item in self.history if item[0] == winner_label]
        avg_conf = float(np.mean(winner_confs)) if winner_confs else conf

        return winner_label, avg_conf

    def reset_history(self):
        """Clears the temporal smoothing history."""
        self.history.clear()
