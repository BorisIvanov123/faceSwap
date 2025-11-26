"""
face_landmarks.py

Processes 106-point landmarks from InsightFace detection.

Consumes:
  - FaceDetectionResult (from face_detection.py)

Produces:
  - LandmarkResult with:
      * landmarks (106, 2)
      * roll angle (deg)
      * face center
      * face width/height
      * bbox
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional
from math import atan2, degrees


# ============================
# DATA CLASS
# ============================

@dataclass
class LandmarkResult:
    landmarks: np.ndarray          # (106, 2) float32
    roll_angle: float
    center_point: tuple            # (x, y)
    face_width: float
    face_height: float
    bbox: tuple                    # (x1, y1, x2, y2)


# ============================
# MAIN PROCESSOR
# ============================

class FaceLandmarkProcessor:
    def __init__(self):
        pass

    def process(self, detection_result, img_shape=None) -> Optional[LandmarkResult]:
        """
        Input:
          - detection_result: FaceDetectionResult (or None)

        Returns:
          - LandmarkResult or None
        """
        if detection_result is None:
            return None

        lm = detection_result.landmarks_2d.astype(np.float32)

        if lm.shape != (106, 2):
            raise ValueError(f"Expected (106,2) landmarks but got {lm.shape}")

        x1, y1, x2, y2 = detection_result.bbox
        face_width = float(x2 - x1)
        face_height = float(y2 - y1)
        center_x = float(x1 + face_width / 2.0)
        center_y = float(y1 + face_height / 2.0)

        roll = self._estimate_roll_angle(lm)

        return LandmarkResult(
            landmarks=lm,
            roll_angle=roll,
            center_point=(center_x, center_y),
            face_width=face_width,
            face_height=face_height,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
        )

    # -----------------------------------

    def _estimate_roll_angle(self, landmarks: np.ndarray) -> float:
        """
        Estimate head roll using outer eye corners.

        Uses indices from InsightFace's 106-landmark layout.
        """
        left_eye_idx = 33
        right_eye_idx = 46

        left = landmarks[left_eye_idx]
        right = landmarks[right_eye_idx]

        dx = right[0] - left[0]
        dy = right[1] - left[1]

        angle = degrees(atan2(dy, dx))
        return angle

    # -----------------------------------

    def draw_landmarks(self, img: np.ndarray, landmarks: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
        """
        Debug utility: draw landmarks on an image.
        """
        out = img.copy()
        for (x, y) in landmarks.astype(int):
            cv2.circle(out, (x, y), 1, color, -1)
        return out

    # -----------------------------------

    def get_face_mask_region(self, landmark_result: LandmarkResult, scale: float = 1.2):
        """
        Returns an expanded bounding box region around the face
        for mask generation / inpainting.
        """
        x1, y1, x2, y2 = landmark_result.bbox

        w = x2 - x1
        h = y2 - y1

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        new_w = int(w * scale)
        new_h = int(h * scale)

        x1_new = max(0, cx - new_w // 2)
        y1_new = max(0, cy - new_h // 2)
        x2_new = cx + new_w // 2
        y2_new = cy + new_h // 2

        return (x1_new, y1_new, x2_new, y2_new)
