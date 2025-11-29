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
    def __init__(self, ctx_id=0):
        """
        Uses the built-in parsing model included in buffalo_l.
        NO DOWNLOADS REQUIRED.
        """
        self.app = FaceAnalysis(
            name="buffalo_l",      # ← important change
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=ctx_id)

    def parse(self, img_bgr):
        if img_bgr is None:
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        outputs = self.app.get(img_rgb)

        # InsightFace returns parsing mask here:
        if not outputs or "parsing" not in outputs[0]:
            return None

        seg = outputs[0]["parsing"].astype(np.uint8)

        hair  = (seg == 17).astype(np.uint8) * 255
        skin  = (seg == 1).astype(np.uint8) * 255
        eyes  = np.isin(seg, [4, 5, 12, 13]).astype(np.uint8) * 255
        mouth = np.isin(seg, [10, 11]).astype(np.uint8) * 255
        face  = np.isin(seg, [1, 8, 9]).astype(np.uint8) * 255

        return FaceParsingResult(
            seg_map=seg,
            hair_mask=hair,
            skin_mask=skin,
            eye_mask=eyes,
            mouth_mask=mouth,
            face_mask=face,
            processed_img=img_rgb
        )

    def visualize_masks(self, result):
        img_bgr = cv2.cvtColor(result.processed_img, cv2.COLOR_RGB2BGR)
        overlay = img_bgr.copy()

        overlay[result.hair_mask > 0]  = (0, 0, 255)
        overlay[result.skin_mask > 0]  = (0, 255, 0)
        overlay[result.eye_mask > 0]   = (255, 0, 0)
        overlay[result.mouth_mask > 0] = (0, 255, 255)

        return overlay
