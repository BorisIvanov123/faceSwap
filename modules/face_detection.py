"""
Face Detection using InsightFace (RetinaFace + ArcFace alignment)
Landmark-aware expanded crop (fixes missing hair detection)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from insightface.app import FaceAnalysis

# ============================
# CONFIG
# ============================

DEFAULT_DET_SIZE = (640, 640)
DEFAULT_MIN_SCORE = 0.30


# ============================
# DATA STRUCTURE
# ============================

@dataclass
class FaceDetectionResult:
    bbox: np.ndarray             # [x1, y1, x2, y2]
    detection_score: float
    landmarks_2d: np.ndarray     # shape (106, 2)
    aligned_face: np.ndarray     # 112×112 aligned BGR
    original_face: np.ndarray    # tight crop
    expanded_face: np.ndarray    # hair + full head crop


# ============================
# FACE DETECTOR
# ============================

class FaceDetector:
    def __init__(self, model_name="buffalo_l", det_size=DEFAULT_DET_SIZE, ctx_id=0):
        """
        model_name: insightface model pack directory (buffalo_l)
        ctx_id: GPU index (0), or -1 for CPU
        """
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.det_size = det_size

    # -----------------------------------

    def detect_faces(self, img_bgr, min_score=DEFAULT_MIN_SCORE):
        if img_bgr is None:
            raise ValueError("Input image is None.")

        faces = self.app.get(img_bgr)

        if len(faces) == 0:
            return None

        # Highest-scoring face
        best = max(faces, key=lambda f: f.det_score)

        if best.det_score < min_score:
            return None

        # Get aligned 112×112 crop
        aligned = self._extract_aligned_face(img_bgr, best)

        # Original tight crop
        orig_crop = self._extract_bbox_crop(img_bgr, best.bbox)

        # Expanded crop (landmark-aware)
        expanded_crop = self._extract_expanded_crop(
            img_bgr,
            best.bbox,
            best.landmark_2d_106
        )

        return FaceDetectionResult(
            bbox=best.bbox.astype(int),
            detection_score=best.det_score,
            landmarks_2d=best.landmark_2d_106,
            aligned_face=aligned,
            original_face=orig_crop,
            expanded_face=expanded_crop
        )

    # -----------------------------------

    def _extract_bbox_crop(self, img, bbox):
        x1, y1, x2, y2 = bbox.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        return img[y1:y2, x1:x2].copy()

    # -----------------------------------
    # 🔥 LANDMARK-AWARE EXPANDED CROP
    # -----------------------------------

    def _extract_expanded_crop(self, img, bbox, landmarks,
                               side_expand_ratio=0.25,
                               bottom_expand_ratio=0.20):
        """
        Expand bbox using forehead height estimated from facial landmarks.
        Much more reliable than a fixed top_expand percentage.

        Uses eyebrow → chin distance to estimate how far upward we must expand
        to capture the full hair region (works for kids too).
        """

        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)

        face_width = x2 - x1
        face_height = y2 - y1

        # === 1. Choose landmark indices ===
        # Approx landmarks from InsightFace 106 layout:
        L_EYEBROW = 70
        R_EYEBROW = 76
        CHIN = 90

        # === 2. Eyebrow & chin centers ===
        eyebrow_y = int((landmarks[L_EYEBROW][1] + landmarks[R_EYEBROW][1]) / 2)
        chin_y = int(landmarks[CHIN][1])

        # === 3. Forehead estimation ===
        eyebrow_to_chin = chin_y - eyebrow_y
        forehead_height = int(eyebrow_to_chin * 0.60)  # Best universal ratio

        # === 4. Compute new expanded box ===
        new_y1 = max(0, y1 - forehead_height)
        new_y2 = min(h, int(y2 + face_height * bottom_expand_ratio))

        new_x1 = max(0, int(x1 - face_width * side_expand_ratio))
        new_x2 = min(w, int(x2 + face_width * side_expand_ratio))

        # === 5. Crop ===
        return img[new_y1:new_y2, new_x1:new_x2].copy()

    # -----------------------------------

    def _extract_aligned_face(self, img, face_obj):
        """
        Reconstruct a proper 112x112 ArcFace-aligned face
        using 5-point alignment derived from InsightFace 106 landmarks.
        """
        lm5 = face_obj.landmark_2d_106[
            [33, 46, 60, 72, 76], :
        ].astype(np.float32)

        ref5 = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)

        M, _ = cv2.estimateAffinePartial2D(lm5, ref5, method=cv2.LMEDS)
        aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0)

        return aligned


# ============================
# UTILITY
# ============================

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


# ============================
# MANUAL TEST
# ============================

if __name__ == "__main__":
    import os

    TEST_IMG = "photos/faces/test.jpg"
    OUTPUT = "photos/output_faces/test_aligned.png"

    os.makedirs("photos/output_faces", exist_ok=True)

    img = load_image(TEST_IMG)
    detector = FaceDetector(ctx_id=0)

    result = detector.detect_faces(img)

    if result is None:
        print("❌ No face detected")
    else:
        print("✔ Face detected!")
        print("BBox:", result.bbox)
        print("Score:", result.detection_score)

        cv2.imwrite(OUTPUT, result.aligned_face)
        print("Saved:", OUTPUT)
