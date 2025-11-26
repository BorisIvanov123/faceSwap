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

Used by:
  - appearance_extraction.py
  - mask_generator.py
  - quality_control
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
    hair_mask: np.ndarray          # (H, W) uint8 (0/255)
    skin_mask: np.ndarray
    eye_mask: np.ndarray
    mouth_mask: np.ndarray
    face_mask: np.ndarray          # skin + ears
    processed_img: np.ndarray      # RGB image used for parsing


# ============================
# MODEL WRAPPER
# ============================

class BiSeNetModel(torch.nn.Module):
    """
    Wrapper around BiSeNet Face Parsing (19 classes).
    Expects the user to provide the BiSeNet implementation.
    """

    def __init__(self, n_classes=19):
        super().__init__()
        from model.BiSeNet import BiSeNet     # <-- must exist in your repo
        self.net = BiSeSeNet(n_classes=n_classes)
        self.n_classes = n_classes

    def load_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"BiSeNet checkpoint missing: {ckpt_path}")

        data = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in data:
            self.net.load_state_dict(data["state_dict"], strict=False)
        else:
            # some checkpoints store the raw state dict
            self.net.load_state_dict(data, strict=False)

        self.net.eval()

    def forward(self, x):
        # BiSeNet returns (out, feat16, feat32)
        out, _, _ = self.net(x)
        return out    # logits (B, C, H, W)


# ============================
# FACE PARSER
# ============================

class FaceParser:
    """
    Performs facial semantic segmentation using BiSeNet.
    """

    def __init__(self, model_path="models/bisenet/bisenet.pth", device="cpu"):
        self.device = torch.device(device)

        # Load model
        self.model = BiSeNetModel(n_classes=19)
        self.model.load_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Mean/std for CelebAMask-HQ model
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Required input size for parsing
        self.input_size = (512, 512)

    # -------------------------------------------------------
    def parse(self, face_img_bgr) -> Optional[FaceParsingResult]:
        """
        Run parsing on a face crop (BGR).
        Returns masks in original crop resolution.
        """
        if face_img_bgr is None:
            return None

        original_h, original_w = face_img_bgr.shape[:2]

        # Convert to RGB
        img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to model input
        resized = cv2.resize(img_rgb, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Normalize
        tensor = self._preprocess(resized)
        tensor = tensor.to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(tensor)[0]      # (C, H, W)

        seg_small = logits.argmax(0).cpu().numpy().astype(np.uint8)

        # Upscale segmentation to original face crop size
        seg = cv2.resize(
            seg_small,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )

        # Build individual masks
        masks = self._make_masks(seg)

        return FaceParsingResult(
            seg_map=seg,
            processed_img=img_rgb,
            **masks
        )

    # -------------------------------------------------------
    def _preprocess(self, img_rgb):
        """
        Convert RGB image to normalized CHW tensor for BiSeNet.
        """
        img_float = img_rgb.astype(np.float32) / 255.0
        img_norm = (img_float - self.mean) / self.std
        img_chw = img_norm.transpose(2, 0, 1)
        return torch.from_numpy(img_chw).unsqueeze(0)

    # -------------------------------------------------------
    def _make_masks(self, seg):
        """
        Create binary masks according to CelebAMask-HQ label indices.
        Label map (commonly used):
            0 background
            1 skin
            2 nose
            3 eyes_glass
            4 left eye
            5 right eye
            6 left brow
            7 right brow
            8 left ear
            9 right ear
            10 mouth outer
            11 mouth inner
            12 left iris
            13 right iris
            17 hair
        """

        hair_mask  = (seg == 17).astype(np.uint8) * 255
        skin_mask  = (seg == 1).astype(np.uint8) * 255
        eye_mask   = ((seg == 4) | (seg == 5) | (seg == 12) | (seg == 13)).astype(np.uint8) * 255
        mouth_mask = ((seg == 10) | (seg == 11)).astype(np.uint8) * 255
        face_mask  = ((seg == 1) | (seg == 8) | (seg == 9)).astype(np.uint8) * 255  # skin + ears

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
        Overlay masks for debugging.
        Returns BGR visualization.
        """
        img = cv2.cvtColor(parsing_result.processed_img, cv2.COLOR_RGB2BGR)
        overlay = img.copy()

        overlay[parsing_result.hair_mask > 0]  = (0, 0, 255)
        overlay[parsing_result.skin_mask > 0]  = (0, 255, 0)
        overlay[parsing_result.eye_mask > 0]   = (255, 0, 0)
        overlay[parsing_result.mouth_mask > 0] = (0, 255, 255)

        return overlay
