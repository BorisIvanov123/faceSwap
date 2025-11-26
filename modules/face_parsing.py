"""
face_parsing.py
-------------------
Performs semantic segmentation on the aligned/original face image.

Model:
  - BiSeNet (face parsing) with 19–20 facial regions
    (hair, skin, eyes, eyebrows, lips, nose, etc.)

Consumes:
  - FaceDetectionResult (original_face crop or aligned_face crop)

Produces:
  - Pixel-level masks for:
      hair_mask
      skin_mask
      eye_mask
      mouth_mask
      face_mask (skin + ears)
  - Raw segmentation map (int label per pixel)

Used later for:
  - appearance_extraction.py
  - mask_generator.py (inpainting masks)
  - QC consistency
"""

import os
import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional


# ----------------------------
# DATA STRUCTURE
# ----------------------------

@dataclass
class FaceParsingResult:
    seg_map: np.ndarray            # shape (H, W), values 0..num_classes
    hair_mask: np.ndarray          # binary mask (H, W)
    skin_mask: np.ndarray          # binary mask
    eye_mask: np.ndarray           # binary mask
    mouth_mask: np.ndarray         # binary mask
    face_mask: np.ndarray          # skin + ears regions
    processed_img: np.ndarray      # the image passed to segmentation


# ----------------------------
# MODEL LOADER
# ----------------------------

class BiSeNetModel(torch.nn.Module):
    """
    Wrapper around BiSeNet for face parsing.
    We assume you have the pretrained checkpoint downloaded.
    """

    def __init__(self, n_classes=19):
        super().__init__()
        from model.BiSeNet import BiSeNet  # You must include BiSeNet in your repo
        self.net = BiSeNet(n_classes=n_classes)
        self.n_classes = n_classes

    def load(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"BiSeNet checkpoint not found: {ckpt_path}")
        data = torch.load(ckpt_path, map_location="cpu")
        self.net.load_state_dict(data["state_dict"])
        self.net.eval()

    def forward(self, x):
        return self.net(x)[0]    # output logits (B, C, H, W)


# ----------------------------
# MAIN PARSER CLASS
# ----------------------------

class FaceParser:
    def __init__(self, model_path="models/bisenet/bisenet.pth", device="cpu"):
        """
        Params:
          model_path: pretrained BiSeNet checkpoint file
          device: "cpu" or "cuda"
        """
        self.device = torch.device(device)

        self.model = BiSeNetModel(n_classes=19)
        self.model.load(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Mean/std for BiSeNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    # -------------------------------------------------------
    def parse(self, face_img_bgr) -> Optional[FaceParsingResult]:
        """
        Run parsing on a face image (BGR).
        Should be a tight crop of the face (from detection).
        """
        if face_img_bgr is None:
            return None

        h, w = face_img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0

        # Normalize
        img_norm = (img_float - self.mean) / self.std
        img_norm = img_norm.transpose(2, 0, 1)  # CHW

        tensor = torch.from_numpy(img_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)[0]  # (C, H, W)

        seg = logits.argmax(0).cpu().numpy().astype(np.uint8)

        # Build masks
        hair_mask = (seg == 17).astype(np.uint8) * 255
        skin_mask = (seg == 1).astype(np.uint8) * 255       # usually label 1
        eye_mask = ((seg == 4) | (seg == 5)).astype(np.uint8) * 255
        mouth_mask = ((seg == 11) | (seg == 12) | (seg == 13)).astype(np.uint8) * 255
        face_mask = ((seg == 1) | (seg == 2)).astype(np.uint8) * 255  # skin + ears

        return FaceParsingResult(
            seg_map=seg,
            hair_mask=hair_mask,
            skin_mask=skin_mask,
            eye_mask=eye_mask,
            mouth_mask=mouth_mask,
            face_mask=face_mask,
            processed_img=face_img_bgr
        )

    # -------------------------------------------------------
    def visualize_masks(self, parsing_result: FaceParsingResult):
        """
        Overlay masks for quick debugging.
        """
        img = parsing_result.processed_img.copy()

        overlay = img.copy()
        overlay[parsing_result.hair_mask > 0] = (0, 0, 255)     # red hair
        overlay[parsing_result.skin_mask > 0] = (0, 255, 0)     # green skin
        overlay[parsing_result.eye_mask > 0] = (255, 0, 0)      # blue eyes
        overlay[parsing_result.mouth_mask > 0] = (255, 255, 0)  # yellow mouth

        return overlay
