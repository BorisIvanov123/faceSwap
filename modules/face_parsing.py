"""
Face parsing using InsightFace built-in BiseNet-ONNX model.
Fast, lightweight, no PyTorch checkpoint needed.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from insightface.app import FaceAnalysis


@dataclass
class FaceParsingResult:
    seg_map: np.ndarray
    hair_mask: np.ndarray
    skin_mask: np.ndarray
    eye_mask: np.ndarray
    mouth_mask: np.ndarray
    face_mask: np.ndarray
    processed_img: np.ndarray


class FaceParser:
    def __init__(self, ctx_id=-1):
        """
        ctx_id = -1 → CPU
        ctx_id = 0  → GPU
        Uses InsightFace's built-in parsing model.
        """
        self.app = FaceAnalysis(
            name="parsing_bisenet",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=ctx_id)

    def parse(self, img_bgr):
        if img_bgr is None:
            return None

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # InsightFace returns segmentation in original resolution
        parsing = self.app.get_parsing(img_rgb)  # (H, W) int labels
        seg = parsing.astype(np.uint8)

        # Build masks
        masks = self._make_masks(seg)

        return FaceParsingResult(
            seg_map=seg,
            processed_img=img_rgb,
            **masks
        )

    def _make_masks(self, seg):
        hair_mask  = (seg == 17).astype(np.uint8) * 255
        skin_mask  = (seg == 1).astype(np.uint8) * 255
        eye_mask   = ((seg == 4) | (seg == 5) | (seg == 12) | (seg == 13)).astype(np.uint8) * 255
        mouth_mask = ((seg == 10) | (seg == 11)).astype(np.uint8) * 255
        face_mask  = ((seg == 1) | (seg == 8) | (seg == 9)).astype(np.uint8) * 255

        return dict(
            hair_mask=hair_mask,
            skin_mask=skin_mask,
            eye_mask=eye_mask,
            mouth_mask=mouth_mask,
            face_mask=face_mask
        )

    def visualize_masks(self, result: FaceParsingResult):
        img_bgr = cv2.cvtColor(result.processed_img, cv2.COLOR_RGB2BGR)
        overlay = img_bgr.copy()

        overlay[result.hair_mask > 0]  = (0, 0, 255)
        overlay[result.skin_mask > 0]  = (0, 255, 0)
        overlay[result.eye_mask > 0]   = (255, 0, 0)
        overlay[result.mouth_mask > 0] = (0, 255, 255)

        return overlay
