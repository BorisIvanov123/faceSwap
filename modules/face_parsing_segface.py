"""
face_parsing_segface.py - SegFace Face Parsing
State-of-the-art face parsing using SegFace (AAAI 2025)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

# Add SegFace to path
sys.path.insert(0, '/workspace/faceSwap/SegFace')
from network import get_model

# SegFace CelebAMask-HQ label order (different from BiSeNet!)
SEGFACE_LABELS = {
    0: 'background',
    1: 'neck',
    2: 'skin',
    3: 'cloth',
    4: 'l_ear',
    5: 'r_ear',
    6: 'l_brow',
    7: 'r_brow',
    8: 'l_eye',
    9: 'r_eye',
    10: 'nose',
    11: 'mouth',
    12: 'l_lip',
    13: 'u_lip',
    14: 'hair',
    15: 'eye_g',
    16: 'hat',
    17: 'ear_r',
    18: 'neck_l',
}

# Mapping for our masks
HAIR_LABELS = [14]
SKIN_LABELS = [2]
EYE_LABELS = [8, 9]
MOUTH_LABELS = [11, 12, 13]
FACE_LABELS = [2, 6, 7, 8, 9, 10, 11, 12, 13]  # skin, brows, eyes, nose, mouth, lips


@dataclass
class FaceParsingResult:
    seg_map: np.ndarray
    hair_mask: np.ndarray
    skin_mask: np.ndarray
    eye_mask: np.ndarray
    mouth_mask: np.ndarray
    face_mask: np.ndarray
    processed_img: np.ndarray


class FaceParserSegFace:
    def __init__(self, model_path: str = "SegFace/weights/mobilenet_celeba_512/model_299.pt", 
                 backbone: str = "mobilenet",
                 input_resolution: int = 512,
                 device: str = None):
        
        self.input_resolution = input_resolution
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = get_model("segface_celeb", input_resolution, backbone)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict_backbone'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        print(f"SegFace loaded: {backbone} backbone, {input_resolution}x{input_resolution}")

    def parse(self, img_bgr: np.ndarray) -> Optional[FaceParsingResult]:
        if img_bgr is None or img_bgr.size == 0:
            return None

        original_size = img_bgr.shape[:2]  # H, W
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize to model input
        img_resized = cv2.resize(img_rgb, (self.input_resolution, self.input_resolution))
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        img_tensor = (img_tensor - self.mean) / self.std
        
        # Dummy labels and dataset (required by model but not used for inference)
        dummy_labels = {"segmentation": torch.zeros(1, self.input_resolution, self.input_resolution).to(self.device)}
        dummy_dataset = torch.tensor([0]).to(self.device)
        
        # Run inference
        with torch.no_grad():
            seg_output = self.model(img_tensor, dummy_labels, dummy_dataset)
            
            # Interpolate to input resolution if needed
            seg_output = F.interpolate(seg_output, size=(self.input_resolution, self.input_resolution), 
                                       mode='bilinear', align_corners=False)
            
            # Get prediction
            seg_map = seg_output.softmax(dim=1).argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
        # Resize back to original size
        seg_map = cv2.resize(seg_map, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Resize processed image back
        processed_rgb = cv2.resize(img_rgb, (original_size[1], original_size[0]))
        
        return FaceParsingResult(
            seg_map=seg_map,
            hair_mask=(seg_map == 14).astype(np.uint8) * 255,
            skin_mask=(seg_map == 2).astype(np.uint8) * 255,
            eye_mask=((seg_map == 8) | (seg_map == 9)).astype(np.uint8) * 255,
            mouth_mask=((seg_map == 11) | (seg_map == 12) | (seg_map == 13)).astype(np.uint8) * 255,
            face_mask=np.isin(seg_map, FACE_LABELS).astype(np.uint8) * 255,
            processed_img=processed_rgb,
        )

    def visualize_all_labels(self, result: FaceParsingResult) -> np.ndarray:
        """Visualize ALL 19 labels with distinct colors."""
        # BGR colors matching SegFace visualize.py
        colors = np.array([
            [0, 0, 0],        # 0: background
            [0, 64, 128],     # 1: neck
            [80, 80, 200],    # 2: skin
            [0, 192, 0],      # 3: cloth
            [0, 0, 64],       # 4: l_ear
            [0, 0, 192],      # 5: r_ear
            [128, 128, 0],    # 6: l_brow
            [128, 128, 128],  # 7: r_brow
            [128, 0, 0],      # 8: l_eye
            [128, 0, 128],    # 9: r_eye
            [0, 128, 0],      # 10: nose
            [0, 128, 64],     # 11: mouth
            [128, 0, 64],     # 12: l_lip
            [0, 128, 192],    # 13: u_lip
            [128, 0, 192],    # 14: hair - PURPLE
            [0, 128, 128],    # 15: eye_g
            [128, 128, 64],   # 16: hat
            [128, 128, 192],  # 17: ear_r
            [0, 64, 0],       # 18: neck_l
        ], dtype=np.uint8)
        
        img_bgr = cv2.cvtColor(result.processed_img, cv2.COLOR_RGB2BGR)
        overlay = colors[result.seg_map]
        
        return cv2.addWeighted(img_bgr, 0.5, overlay, 0.5, 0)

    def visualize_masks(self, result: FaceParsingResult, alpha: float = 0.5) -> np.ndarray:
        """Visualize key masks with distinct colors."""
        img_bgr = cv2.cvtColor(result.processed_img, cv2.COLOR_RGB2BGR)
        overlay = img_bgr.copy()
        
        overlay[result.hair_mask > 0] = [128, 0, 128]    # Purple
        overlay[result.skin_mask > 0] = [0, 255, 0]      # Green
        overlay[result.eye_mask > 0] = [255, 0, 0]       # Blue
        overlay[result.mouth_mask > 0] = [0, 255, 255]   # Yellow
        
        return cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)
