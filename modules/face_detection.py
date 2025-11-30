"""
Face Detection using InsightFace (RetinaFace + ArcFace alignment)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from insightface.app import FaceAnalysis


DEFAULT_DET_SIZE = (640, 640)
DEFAULT_MIN_SCORE = 0.30


@dataclass
class FaceDetectionResult:
    bbox: np.ndarray
    detection_score: float
    landmarks_2d: np.ndarray
    aligned_face: np.ndarray
    original_face: np.ndarray
    expanded_face: np.ndarray  # NEW: expanded crop for hair


class FaceDetector:
    def __init__(self, model_name="buffalo_l", det_size=DEFAULT_DET_SIZE, ctx_id=0):
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.det_size = det_size

    def detect_faces(self, img_bgr, min_score=DEFAULT_MIN_SCORE):
        if img_bgr is None:
            raise ValueError("Input image is None.")

        faces = self.app.get(img_bgr)

        if len(faces) == 0:
            return None

        best = max(faces, key=lambda f: f.det_score)

        if best.det_score < min_score:
            return None

        aligned = self._extract_aligned_face(img_bgr, best)
        orig_crop = self._extract_bbox_crop(img_bgr, best.bbox)
        expanded_crop = self._extract_expanded_crop(img_bgr, best.bbox)

        return FaceDetectionResult(
            bbox=best.bbox.astype(int),
            detection_score=best.det_score,
            landmarks_2d=best.landmark_2d_106,
            aligned_face=aligned,
            original_face=orig_crop,
            expanded_face=expanded_crop
        )

    def _extract_bbox_crop(self, img, bbox):
        x1, y1, x2, y2 = bbox.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        return img[y1:y2, x1:x2].copy()

    def _extract_expanded_crop(self, img, bbox, top_expand=0.8, side_expand=0.3, bottom_expand=0.15):
        """Expand bbox to include hair region."""
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = img.shape[:2]

        face_width = x2 - x1
        face_height = y2 - y1

        new_x1 = max(0, int(x1 - face_width * side_expand))
        new_x2 = min(w, int(x2 + face_width * side_expand))
        new_y1 = max(0, int(y1 - face_height * top_expand))
        new_y2 = min(h, int(y2 + face_height * bottom_expand))

        return img[new_y1:new_y2, new_x1:new_x2].copy()

    def _extract_aligned_face(self, img, face_obj):
        lm5 = face_obj.landmark_2d_106[[33, 46, 60, 72, 76], :].astype(np.float32)

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


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img