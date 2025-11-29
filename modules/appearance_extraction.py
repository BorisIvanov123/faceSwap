"""
appearance_extraction.py
-------------------------
Extract high-level appearance attributes from face parsing output.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class AppearanceResult:
    hair_color_rgb: tuple
    hair_color_hex: str
    hair_shade: str           # "black" | "dark brown" | "brown" | "light brown" | "blonde" | "red" | "gray/white"

    skin_color_rgb: tuple
    skin_tone: str            # "light" | "medium" | "tan" | "dark"

    eye_color_rgb: tuple
    eye_color_name: str       # "brown" | "blue" | "green" | "hazel" | "gray"

    meta: dict


def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))


def classify_hair_color(rgb):
    """
    More accurate hair color classification using luminance and color ratios.
    """
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Very dark = black
    if lum < 50:
        return "black"

    # Very light
    if lum > 200:
        if r > g and r > b:
            return "blonde"
        return "gray/white"

    # Check for red/auburn (red channel dominant)
    if r > g * 1.3 and r > b * 1.3:
        if lum > 120:
            return "auburn"
        return "red"

    # Brown shades based on luminance
    if lum < 80:
        return "dark brown"
    elif lum < 120:
        return "brown"
    else:
        return "light brown"


def classify_skin_tone(rgb):
    """Skin tone classification."""
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    if lum > 200:
        return "very light"
    elif lum > 170:
        return "light"
    elif lum > 130:
        return "medium"
    elif lum > 90:
        return "tan"
    else:
        return "dark"


def classify_eye_color(rgb):
    """Eye color classification using HSV for better accuracy."""
    r, g, b = rgb

    # Convert to HSV
    pixel_bgr = np.array([[[b, g, r]]], dtype=np.uint8)
    pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = pixel_hsv[0, 0]

    # Low saturation = gray or very dark
    if s < 25:
        if v < 80:
            return "dark brown"
        return "gray"

    # Hue-based (OpenCV: 0-180)
    if 100 <= h <= 130:
        return "blue"
    elif 35 <= h <= 85:
        return "green"
    elif 15 <= h <= 35:
        return "hazel"

    return "brown"


class AppearanceExtractor:

    def __init__(self, min_pixels: int = 50):
        self.min_pixels = min_pixels

    def extract(self, parsing_result) -> AppearanceResult:
        img_rgb = parsing_result.processed_img

        hair_rgb = self._extract_color_robust(img_rgb, parsing_result.hair_mask)
        skin_rgb = self._extract_color_robust(img_rgb, parsing_result.skin_mask)
        eye_rgb = self._extract_color_robust(img_rgb, parsing_result.eye_mask)

        hair_count = int((parsing_result.hair_mask > 0).sum())
        skin_count = int((parsing_result.skin_mask > 0).sum())
        eye_count = int((parsing_result.eye_mask > 0).sum())

        return AppearanceResult(
            hair_color_rgb=hair_rgb,
            hair_color_hex=rgb_to_hex(hair_rgb),
            hair_shade=classify_hair_color(hair_rgb),

            skin_color_rgb=skin_rgb,
            skin_tone=classify_skin_tone(skin_rgb),

            eye_color_rgb=eye_rgb,
            eye_color_name=classify_eye_color(eye_rgb),

            meta={
                "hair_pixel_count": hair_count,
                "skin_pixel_count": skin_count,
                "eye_pixel_count": eye_count,
            }
        )

    def _extract_color_robust(self, img_rgb: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
        """
        Extract dominant color using median (robust to outliers).
        """
        ys, xs = np.where(mask > 0)

        if len(xs) < self.min_pixels:
            # Fallback
            mean = img_rgb.mean(axis=(0, 1))
            return (int(mean[0]), int(mean[1]), int(mean[2]))

        pixels = img_rgb[ys, xs]

        # Use median for robustness against outliers/edge pixels
        median_rgb = np.median(pixels, axis=0)

        return (int(median_rgb[0]), int(median_rgb[1]), int(median_rgb[2]))

    def save_debug_masks(self, parsing_result, img_bgr: np.ndarray, output_dir: str):
        """
        Save individual mask visualizations for debugging.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Save each mask
        masks = {
            'hair': parsing_result.hair_mask,
            'skin': parsing_result.skin_mask,
            'eye': parsing_result.eye_mask,
            'mouth': parsing_result.mouth_mask,
            'face': parsing_result.face_mask,
        }

        for name, mask in masks.items():
            # Raw mask
            cv2.imwrite(f"{output_dir}/mask_{name}.png", mask)

            # Masked region extraction
            masked = img_bgr.copy()
            masked[mask == 0] = 0
            cv2.imwrite(f"{output_dir}/masked_{name}.png", masked)

        # Full segmentation map colorized
        seg_colors = np.zeros((*parsing_result.seg_map.shape, 3), dtype=np.uint8)
        color_map = {
            0: (0, 0, 0),        # background
            1: (255, 224, 189),  # skin
            2: (255, 0, 255),    # nose
            3: (0, 255, 255),    # eyeglasses
            4: (0, 0, 255),      # l_eye
            5: (0, 0, 255),      # r_eye
            6: (0, 255, 0),      # l_brow
            7: (0, 255, 0),      # r_brow
            8: (255, 165, 0),    # l_ear
            9: (255, 165, 0),    # r_ear
            10: (255, 0, 0),     # mouth
            11: (200, 0, 0),     # u_lip
            12: (150, 0, 0),     # l_lip
            13: (139, 69, 19),   # hair - brown
            14: (128, 0, 128),   # hat
            15: (255, 215, 0),   # earring
            16: (192, 192, 192), # necklace
            17: (210, 180, 140), # neck
            18: (100, 100, 100), # cloth
        }

        for label, color in color_map.items():
            seg_colors[parsing_result.seg_map == label] = color

        cv2.imwrite(f"{output_dir}/segmentation_colored.png", cv2.cvtColor(seg_colors, cv2.COLOR_RGB2BGR))

        print(f"Debug masks saved to {output_dir}/")