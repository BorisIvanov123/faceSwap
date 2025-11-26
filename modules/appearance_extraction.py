"""
appearance_extraction.py
-------------------------
Extract high-level appearance attributes from face parsing output.

Consumes:
  - FaceParsingResult (from face_parsing.py)

Produces:
  - Hair color (RGB + HEX + shade)
  - Skin tone (light / medium / dark)
  - Eye color (coarse classification)
  - Meta statistics (pixel counts)

Used by:
  - prompt_builder.py
  - identity_profile.py
"""

import numpy as np
from dataclasses import dataclass


# ============================
# DATA CLASS
# ============================

@dataclass
class AppearanceResult:
    hair_color_rgb: tuple        # (r, g, b)
    hair_color_hex: str          # "#RRGGBB"
    hair_shade: str              # "light" | "medium" | "dark"

    skin_color_rgb: tuple
    skin_tone: str               # "light" | "medium" | "dark"

    eye_color_rgb: tuple
    eye_color_name: str          # "brown" | "blue" | "green" | "hazel"

    meta: dict                   # misc stats (pixel counts etc.)


# ============================
# HELPERS
# ============================

def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))


def classify_shade(rgb):
    """
    Classify luminance into light / medium / dark.
    Works well for both hair & skin.
    """
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    if lum > 180:
        return "light"
    elif lum > 80:
        return "medium"
    else:
        return "dark"


def closest_eye_color(rgb):
    """
    Light-weight, stable eye color estimator.
    Priority-based classification.
    """
    r, g, b = rgb

    # hazel: warm but not fully brown
    if r > 90 and g > 70 and b < 60:
        return "hazel"

    # blue-ish (strong blue component)
    if b > 100 and g > 80:
        return "blue"

    # green-ish (balanced green & blue)
    if g > 100 and b > 60:
        return "green"

    # fallback
    return "brown"


# ============================
# MAIN CLASS
# ============================

class AppearanceExtractor:

    def __init__(self):
        pass

    # ---------------------------------------------

    def extract(self, parsing_result) -> AppearanceResult:
        """
        Main entry point:
        parsing_result must include:
            - processed_img  (RGB image)
            - hair_mask
            - skin_mask
            - eye_mask
        """

        img_rgb = parsing_result.processed_img

        hair_rgb = self._extract_color(img_rgb, parsing_result.hair_mask)
        skin_rgb = self._extract_color(img_rgb, parsing_result.skin_mask)
        eye_rgb  = self._extract_color(img_rgb, parsing_result.eye_mask)

        return AppearanceResult(
            hair_color_rgb=hair_rgb,
            hair_color_hex=rgb_to_hex(hair_rgb),
            hair_shade=classify_shade(hair_rgb),

            skin_color_rgb=skin_rgb,
            skin_tone=classify_shade(skin_rgb),

            eye_color_rgb=eye_rgb,
            eye_color_name=closest_eye_color(eye_rgb),

            meta={
                "hair_pixel_count": int((parsing_result.hair_mask > 0).sum()),
                "skin_pixel_count": int((parsing_result.skin_mask > 0).sum()),
                "eye_pixel_count": int((parsing_result.eye_mask > 0).sum()),
            }
        )

    # ---------------------------------------------

    def _extract_color(self, img_rgb, mask):
        """
        Extracts the mean RGB value inside a segmentation mask.
        If mask has too few pixels → fallback to global image mean.
        """

        ys, xs = np.where(mask > 0)

        # fallback if mask is empty or nearly empty
        if len(xs) < 10:
            mean = img_rgb.mean(axis=(0, 1))
            r, g, b = mean
            return (int(r), int(g), int(b))

        pixels = img_rgb[ys, xs]
        mean_rgb = pixels.mean(axis=0)

        r, g, b = mean_rgb
        return (int(r), int(g), int(b))
