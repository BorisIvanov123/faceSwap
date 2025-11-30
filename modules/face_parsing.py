"""
face_parsing.py - BiSeNet Face Parsing using ONNX Runtime
Ensures parsing always sees the full head by adding safety padding.
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

    # --------------------------------------------------------
    #                FULL-HEAD GUARANTEE PATCH
    # --------------------------------------------------------
    def _pad_full_head(self, img_bgr):
        """
        Ensures BiSeNet sees the entire head even if the crop is too tight.
        Adds 8% padding (min 15px) on each side using BORDER_REPLICATE.
        """

        h, w = img_bgr.shape[:2]
        pad = max(15, int(min(h, w) * 0.08))  # 8% border

        padded = cv2.copyMakeBorder(
            img_bgr,
            pad, pad, pad, pad,
            borderType=cv2.BORDER_REPLICATE
        )
        return padded

    # --------------------------------------------------------
    def parse(self, img_bgr: np.ndarray) -> Optional[FaceParsingResult]:
        if img_bgr is None or img_bgr.size == 0:
            return None

        # 🔥 Guarantee full head visibility BEFORE parsing
        img_bgr = self._pad_full_head(img_bgr)

        original_size = img_bgr.shape[:2]

        # Convert to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to model input
        img_resized = cv2.resize(img_rgb, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Normalize
        img_float = img_resized.astype(np.float32) / 255.0
        img_norm = (img_float - self.mean) / self.std
        img_nchw = img_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        # Run ONNX
        outputs = self.session.run(None, {self.input_name: img_nchw})

        # Get argmax segmentation map
        seg_map = outputs[0][0].argmax(axis=0).astype(np.uint8)

        # Resize seg_map back to padded crop size
        seg_map = cv2.resize(seg_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

        # Restore processed RGB image to original padded size
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

    # --------------------------------------------------------
    def visualize_masks(self, result: FaceParsingResult, alpha: float = 0.5) -> np.ndarray:
        """Visualize with distinct colors for each class."""
        img_bgr = cv2.cvtColor(result.processed_img, cv2.COLOR_RGB2BGR)

        # Create colored segmentation map
        colors = np.array([
            [0, 0, 0],        # background
            [0, 255, 0],      # skin
            [255, 0, 255],    # nose
            [255, 255, 0],    # eye_g
            [255, 0, 0],      # l_eye
            [255, 0, 0],      # r_eye
            [0, 128, 0],      # l_brow
            [0, 128, 0],      # r_brow
            [0, 165, 255],    # l_ear
            [0, 165, 255],    # r_ear
            [0, 255, 255],    # mouth
            [0, 0, 255],      # u_lip
            [0, 0, 200],      # l_lip
            [128, 0, 128],    # hair
            [128, 128, 0],    # hat
            [0, 215, 255],    # earring
            [192, 192, 192],  # neck_l
            [140, 180, 210],  # neck
            [100, 100, 100],  # cloth
        ], dtype=np.uint8)

        overlay = colors[result.seg_map]
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        return cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)

    # --------------------------------------------------------
    def print_label_stats(self, result: FaceParsingResult):
        """Print pixel counts for each label."""
        print("\nLabel Statistics:")
        for label in sorted(np.unique(result.seg_map)):
            count = (result.seg_map == label).sum()
            name = LABEL_MAP.get(label, '?')
            print(f"  {label:2d} ({name:10s}): {count:,} pixels")
