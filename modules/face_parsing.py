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
SKIN_LABELS = [1]
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
    def __init__(self, model_path: str = "weights/resnet34.onnx", ctx_id: int = 0):
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
            hair_mask=(seg_map == 13).astype(np.uint8) * 255,
            skin_mask=(seg_map == 1).astype(np.uint8) * 255,
            eye_mask=((seg_map == 4) | (seg_map == 5)).astype(np.uint8) * 255,
            mouth_mask=((seg_map == 10) | (seg_map == 11) | (seg_map == 12)).astype(np.uint8) * 255,
            face_mask=np.isin(seg_map, FACE_LABELS).astype(np.uint8) * 255,
            processed_img=processed_rgb,
        )

    def visualize_masks(self, result: FaceParsingResult, alpha: float = 0.5) -> np.ndarray:
        """Visualize with distinct colors - BGR format for OpenCV."""
        img_bgr = cv2.cvtColor(result.processed_img, cv2.COLOR_RGB2BGR)
        overlay = img_bgr.copy()
        
        # Apply colors directly in BGR
        overlay[result.hair_mask > 0] = [128, 0, 128]    # Purple
        overlay[result.skin_mask > 0] = [0, 255, 0]      # Green
        overlay[result.eye_mask > 0] = [255, 0, 0]       # Blue
        overlay[result.mouth_mask > 0] = [0, 255, 255]   # Yellow
        
        return cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)

    def visualize_all_labels(self, result: FaceParsingResult) -> np.ndarray:
        """Visualize ALL 19 labels with distinct colors."""
        # BGR colors for each label
        colors = {
            0: [0, 0, 0],         # background - black
            1: [0, 255, 0],       # skin - green
            2: [255, 0, 255],     # nose - magenta
            3: [255, 255, 0],     # eye_g - cyan
            4: [255, 0, 0],       # l_eye - blue
            5: [255, 0, 0],       # r_eye - blue
            6: [0, 100, 0],       # l_brow - dark green
            7: [0, 100, 0],       # r_brow - dark green
            8: [0, 165, 255],     # l_ear - orange
            9: [0, 165, 255],     # r_ear - orange
            10: [0, 255, 255],    # mouth - yellow
            11: [0, 0, 255],      # u_lip - red
            12: [0, 0, 180],      # l_lip - dark red
            13: [128, 0, 128],    # hair - purple
            14: [128, 128, 0],    # hat - teal
            15: [0, 215, 255],    # earring - gold
            16: [192, 192, 192],  # neck_l - silver
            17: [140, 180, 210],  # neck - tan
            18: [80, 80, 80],     # cloth - gray
        }
        
        img_bgr = cv2.cvtColor(result.processed_img, cv2.COLOR_RGB2BGR)
        overlay = np.zeros_like(img_bgr)
        
        for label, color in colors.items():
            overlay[result.seg_map == label] = color
        
        return cv2.addWeighted(img_bgr, 0.5, overlay, 0.5, 0)
