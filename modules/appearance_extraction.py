"""
appearance_extraction.py
-------------------------
Extract high-level appearance attributes from face parsing output.

Consumes:
  - FaceParsingResult (from face_parsing.py)

Produces:
  - Hair color (as RGB, HEX, and human-readable string)
  - Skin tone (light / medium / dark)
  - Eye color (rough classification)
  - Hair shade (light/medium/dark)
  - Optional: hair length estimate

Used by:
  - prompt_builder.py
  - identity_profile.py
"""

import numpy as np
from dataclasses import dataclass


# ----------------------------
# DATA CLASS
# ----------------------------

@dataclass
class AppearanceResult:
    hair_color_rgb: tuple  # (r, g, b)
    hair_color_hex: str  # "#xxxxxx"
    hair_shade: str  # "light" / "medium" / "dark"

    skin_color_rgb: tuple
    skin_tone: str  # "light" / "medium" / "dark"

    eye_color_rgb: tuple
    eye_color_name: str  # "brown", "blue", etc.

    meta: dict  # extra fields


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b)).upper()


def classify_shade(rgb):
    """
    Light vs medium vs dark hair or skin.
    Based on luminance.
    """
    r, g, b = rgb
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    if luminance > 180:
        return "light"
    elif luminance > 80:
        return "medium"
    else:
        return "dark"


def closest_eye_color(rgb):
    """
    Quick & reliable eye color heuristic:
      - Brown (default)
      - Blue
      - Green
      - Hazel
    """
    r, g, b = rgb

    if r > 80 and g > 60 and b < 50:
        return "hazel"

    if b > 100 and g > 80:
        return "blue"

    if g > 100 and b > 60:
        return "green"

    return "brown"  # strongest prior


# ----------------------------
# MAIN CLASS
# ----------------------------

class AppearanceExtractor:

    def __init__(self):
        pass

    # ----------------------------------------------------
    def extract(self, parsing_result) -> AppearanceResult:
        img = parsing_result.processed_img
        seg_mask = parsing_result.seg_map

        # Extract hair / skin / eye regions
        hair_rgb = self._extract_color(img, parsing_result.hair_mask)
        skin_rgb = self._extract_color(img, parsing_result.skin_mask)
        eye_rgb = self._extract_color(img, parsing_result.eye_mask)

        # Classify attributes
        hair_shade = classify_shade(hair_rgb)
        skin_tone = classify_shade(skin_rgb)
        eye_color_name = closest_eye_color(eye_rgb)

        return AppearanceResult(
            hair_color_rgb=hair_rgb,
            hair_color_hex=rgb_to_hex(hair_rgb),
            hair_shade=hair_shade,

            skin_color_rgb=skin_rgb,
            skin_tone=skin_tone,

            eye_color_rgb=eye_rgb,
            eye_color_name=eye_color_name,

            meta={
                "hair_pixel_count": int((parsing_result.hair_mask > 0).sum()),
                "skin_pixel_count": int((parsing_result.skin_mask > 0).sum()),
                "eye_pixel_count": int((parsing_result.eye_mask > 0).sum()),
            }
        )

    # ----------------------------------------------------
    def _extract_color(self, img_rgb, mask):
        """
        Extract dominant color from masked region.
        Returns RGB tuple.
        """
        ys, xs = np.where(mask > 0)

        if len(xs) < 10:  # too few pixels → fallback average of whole image
            mean = img_rgb.mean(axis=(0, 1))
            return tuple(map(int, mean[::-1]))  # BGR → RGB

        # Get all masked pixels
        pixels = img_rgb[ys, xs]
        mean_bgr = pixels.mean(axis=0).astype(np.uint8)
        r, g, b = int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])
        return (r, g, b)
