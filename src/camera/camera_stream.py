"""
camera_stream.py
----------------
Handles webcam access, frame reading, resizing, and FPS calculation for JARVIS.

Usage:
    from src.camera.camera_stream import CameraStream

    cam = CameraStream()
    cam.start()
    while True:
        frame = cam.read_frame()
        if frame is None:
            break
        # ... process frame ...
    cam.release()
"""

import cv2
import time


class CameraStream:
    """
    Manages a live webcam feed.

    Attributes:
        camera_index (int): Device index of the webcam (0 = default).
        width (int): Desired frame width in pixels.
        height (int): Desired frame height in pixels.
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._cap = None

        # FPS tracking
        self._prev_time = 0.0
        self._fps = 0.0

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> bool:
        """
        Opens the webcam.

        Returns:
            True if the camera opened successfully, False otherwise.
        """
        self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            print(
                f"[CameraStream] ERROR: Could not open camera at index {self.camera_index}."
            )
            print(
                "[CameraStream] Tip: Try camera_index=1 if you have multiple cameras."
            )
            return False

        # Request the desired resolution from the driver
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"[CameraStream] Camera opened — Resolution: {actual_w}x{actual_h} "
            f"(requested {self.width}x{self.height})"
        )
        return True

    def release(self):
        """Releases the webcam and closes any OpenCV windows."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
        cv2.destroyAllWindows()
        print("[CameraStream] Camera released.")

    # ------------------------------------------------------------------ #
    #  Frame reading
    # ------------------------------------------------------------------ #

    def read_frame(self):
        """
        Reads the next frame from the webcam.

        Returns:
            numpy.ndarray of shape (H, W, 3) in BGR colour, or None on failure.
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        success, frame = self._cap.read()
        if not success or frame is None:
            print("[CameraStream] WARNING: Failed to grab frame.")
            return None

        # Flip horizontally so it acts like a mirror (more intuitive for hand gestures)
        frame = cv2.flip(frame, 1)
        return frame

    # ------------------------------------------------------------------ #
    #  FPS
    # ------------------------------------------------------------------ #

    def compute_fps(self) -> float:
        """
        Computes the current frames-per-second.

        Call this once per loop iteration, after read_frame().

        Returns:
            Current FPS as a float.
        """
        current_time = time.time()
        elapsed = current_time - self._prev_time
        self._fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self._prev_time = current_time
        return self._fps

    @property
    def fps(self) -> float:
        """Last computed FPS value."""
        return self._fps

    # ------------------------------------------------------------------ #
    #  Utility overlay
    # ------------------------------------------------------------------ #

    @staticmethod
    def draw_fps(frame, fps: float):
        """
        Draws a live FPS counter onto a frame (top-left corner).

        Args:
            frame: BGR numpy array (modified in-place).
            fps: Current frames-per-second value.
        """
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),    # Green
            2,
            cv2.LINE_AA,
        )
