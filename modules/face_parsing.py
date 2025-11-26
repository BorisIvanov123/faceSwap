"""
face_parsing.py
-------------------
Performs semantic segmentation on a face crop using BiSeNet (CelebAMask-HQ).

Consumes:
  - FaceDetectionResult.original_face   (or aligned_face)

Produces masks for:
  - hair_mask
  - skin_mask
  - eye_mask
  - mouth_mask
  - face_mask  (skin + ears)
Along with:
  - seg_map (raw labels)
  - processed_img (RGB image used internally)
"""

import os
import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional


# ============================
# RESULT STRUCTURE
# ============================

@dataclass
class FaceParsingResult:
    seg_map: np.ndarray            # (H, W) uint8 labels
    hair_mask: np.ndarray          # (H, W) uint8
    skin_mask: np.ndarray
    eye_mask: np.ndarray
    mouth_mask: np.ndarray
    face_mask: np.ndarray
    processed_img: np.ndarray      # RGB image


# ============================
# MODEL WRAPPER
# ============================

class BiSeNetModel(torch.nn.Module):
    """
    A clean wrapper around your BiSeNet implementation.
    """

    def __init__(self, n_classes=19):
        super().__init__()
        from models.bisenet.bisenet import BiSeNet      # <- CORRECT PATH
        self.net = BiSeNet(n_classes=n_classes)
        self.n_classes = n_classes

    def load_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"BiSeNet checkpoint missing: {ckpt_path}")

        data = torch.load(ckpt_path, map_location="cpu")

        # Support for both raw state_dict and wrapped state_dict
        if "state_dict" in data:
            self.net.load_state_dict(data["state_dict"], strict=False)
        else:
            self.net.load_state_dict(data, strict=False)

        self.net.eval()

    def forward(self, x):
        # Your bisenet.py returns a single output tensor (logits)
        out = self.net(x)
        return out    # (B, C, H, W)


# ============================
# FACE PARSER
# ============================

class FaceParser:
    """
    Runs face semantic segmentation using BiSeNet.
    """

    def __init__(self, model_path="models/bisenet/bisenet.pth", device="cpu"):
        self.device = torch.device(device)

        # Load model
        self.model = BiSeNetModel(n_classes=19)
        self.model.load_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()

        # CelebAMask-HQ normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # BiSeNet input size
        self.input_size = (512, 512)

    # -------------------------------------------------------
    def parse(self, face_img_bgr) -> Optional[FaceParsingResult]:
        """
        Run parsing on a face crop (BGR).
        Returns segmentation masks resized to original size.
        """
        if face_img_bgr is None:
            return None

        H, W = face_img_bgr.shape[:2]

        # Convert BGR → RGB
        img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)

        # Resize for model
        resized = cv2.resize(img_rgb, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Preprocess to tensor
        tensor = self._preprocess(resized).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)[0]      # (C, H, W)

        seg_small = logits.argmax(0).cpu().numpy().astype(np.uint8)

        # Restore to original resolution
        seg = cv2.resize(
            seg_small,
            (W, H),
            interpolation=cv2.INTER_NEAREST
        )

        masks = self._make_masks(seg)

        return FaceParsingResult(
            seg_map=seg,
            processed_img=img_rgb,
            **masks
        )

    # -------------------------------------------------------
    def _preprocess(self, img_rgb):
        img = img_rgb.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)      # HWC → CHW
        return torch.from_numpy(img).unsqueeze(0)

    # -------------------------------------------------------
    def _make_masks(self, seg):
        """
        Mask labels based on CelebAMask-HQ:
            1 skin
            4 left eye
            5 right eye
            12 left iris
            13 right iris
            10 mouth outer
            11 mouth inner
            17 hair
            8/9 ears
        """

        hair_mask  = (seg == 17).astype(np.uint8) * 255
        skin_mask  = (seg == 1).astype(np.uint8) * 255

        eye_mask = (
            (seg == 4) | (seg == 5) | (seg == 12) | (seg == 13)
        ).astype(np.uint8) * 255

        mouth_mask = (
            (seg == 10) | (seg == 11)
        ).astype(np.uint8) * 255

        face_mask = (
            (seg == 1) | (seg == 8) | (seg == 9)
        ).astype(np.uint8) * 255

        return dict(
            hair_mask=hair_mask,
            skin_mask=skin_mask,
            eye_mask=eye_mask,
            mouth_mask=mouth_mask,
            face_mask=face_mask
        )

    # -------------------------------------------------------
    def visualize_masks(self, parsing_result: FaceParsingResult):
        """
        Overlay color-coded masks for debugging.
        """
        img_bgr = cv2.cvtColor(parsing_result.processed_img, cv2.COLOR_RGB2BGR)
        overlay = img_bgr.copy()

        overlay[parsing_result.hair_mask > 0]  = (0, 0, 255)       # red
        overlay[parsing_result.skin_mask > 0]  = (0, 255, 0)       # green
        overlay[parsing_result.eye_mask > 0]   = (255, 0, 0)       # blue
        overlay[parsing_result.mouth_mask > 0] = (0, 255, 255)     # yellow

        return overlay
