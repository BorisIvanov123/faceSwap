"""
face_parsing.py - BiSeNet Face Parsing using ONNX Runtime
"""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import onnxruntime as ort

LABEL_MAP = {
    0: 'background', 1: 'skin', 2: 'nose', 3: 'eye_g',
    4: 'l_eye', 5: 'r_eye', 6: 'l_brow', 7: 'r_brow',
    8: 'l_ear', 9: 'r_ear', 10: 'mouth', 11: 'u_lip',
    12: 'l_lip', 13: 'hair', 14: 'hat', 15: 'ear_r',
    16: 'neck_l', 17: 'neck', 18: 'cloth',
}

HAIR_LABELS = [13]
SKIN_LABELS = [1, 17]
EYE_LABELS = [4, 5]
MOUTH_LABELS = [10, 11, 12]
FACE_LABELS = [1, 2, 4, 5, 6, 7, 10, 11, 12]


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
    def __init__(self, model_path: str = "weights/resnet18.onnx", ctx_id: int = 0):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id >= 0 else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (512, 512)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def parse(self, img_bgr: np.ndarray) -> Optional[FaceParsingResult]:
        if img_bgr is None or img_bgr.size == 0:
            return None

        original_size = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.input_size, interpolation=cv2.INTER_LINEAR)

        img_float = img_resized.astype(np.float32) / 255.0
        img_norm = (img_float - self.mean) / self.std
        img_nchw = img_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: img_nchw})

        seg_map = outputs[0][0].argmax(axis=0).astype(np.uint8)
        seg_map = cv2.resize(seg_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

        processed_rgb = cv2.resize(img_rgb, (original_size[1], original_size[0]))

        return FaceParsingResult(
            seg_map=seg_map,
            hair_mask=(np.isin(seg_map, HAIR_LABELS).astype(np.uint8) * 255),
            skin_mask=(np.isin(seg_map, SKIN_LABELS).astype(np.uint8) * 255),
            eye_mask=(np.isin(seg_map, EYE_LABELS).astype(np.uint8) * 255),
            mouth_mask=(np.isin(seg_map, MOUTH_LABELS).astype(np.uint8) * 255),
            face_mask=(np.isin(seg_map, FACE_LABELS).astype(np.uint8) * 255),
            processed_img=processed_rgb,
        )

    def visualize_masks(self, result: FaceParsingResult, alpha: float = 0.5) -> np.ndarray:
        img_bgr = cv2.cvtColor(result.processed_img, cv2.COLOR_RGB2BGR)
        overlay = img_bgr.copy()
        overlay[result.hair_mask > 0] = (0, 0, 255)
        overlay[result.skin_mask > 0] = (0, 255, 0)
        overlay[result.eye_mask > 0] = (255, 0, 0)
        overlay[result.mouth_mask > 0] = (0, 255, 255)
        return cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)