import os
import json
import numpy as np
import cv2
from dataclasses import dataclass, asdict


@dataclass
class IdentityProfile:
    # ---------------------------
    # CORE IDENTITY
    # ---------------------------
    embedding: np.ndarray  # ArcFace 512-d embedding
    embedding_norm: float  # For consistency validation

    # ---------------------------
    # APPEARANCE ATTRIBUTES
    # ---------------------------
    hair_color_rgb: tuple
    hair_color_hex: str
    hair_shade: str

    skin_color_rgb: tuple
    skin_tone: str

    eye_color_rgb: tuple
    eye_color_name: str

    # ---------------------------
    # FACE MASKS / PARSING
    # ---------------------------
    mask_face: np.ndarray
    mask_skin: np.ndarray
    mask_hair: np.ndarray
    mask_eye: np.ndarray
    mask_mouth: np.ndarray

    seg_map: np.ndarray

    # ---------------------------
    # FACE CROPS
    # ---------------------------
    aligned_face: np.ndarray  # normalized crop for embedding
    expanded_face: np.ndarray  # good for parsing/masking
    original_face: np.ndarray  # raw crop

    # ---------------------------
    # LANDMARKS
    # ---------------------------
    landmarks: np.ndarray
    roll_angle: float

    # ---------------------------
    # META
    # ---------------------------
    pixel_stats: dict
    debug_paths: dict

from .appearance_extraction import AppearanceExtractor

def build_identity_profile(det, lm, emb, parse, appearance, debug_paths=None):
    """
    Combine all extracted modules into a single IdentityProfile object.
    """

    profile = IdentityProfile(
        # Embedding
        embedding=emb.embedding,
        embedding_norm=emb.norm,

        # Appearance
        hair_color_rgb=appearance.hair_color_rgb,
        hair_color_hex=appearance.hair_color_hex,
        hair_shade=appearance.hair_shade,

        skin_color_rgb=appearance.skin_color_rgb,
        skin_tone=appearance.skin_tone,

        eye_color_rgb=appearance.eye_color_rgb,
        eye_color_name=appearance.eye_color_name,

        # Masks
        mask_face=parse.face_mask,
        mask_skin=parse.skin_mask,
        mask_hair=parse.hair_mask,
        mask_eye=parse.eye_mask,
        mask_mouth=parse.mouth_mask,

        seg_map=parse.seg_map,

        # Crops
        aligned_face=det.aligned_face,
        expanded_face=det.expanded_face,
        original_face=det.original_face,

        # Landmarks
        landmarks=lm.landmarks,
        roll_angle=lm.roll_angle,

        # Meta
        pixel_stats={
            "hair": appearance.meta["hair_pixel_count"],
            "skin": appearance.meta["skin_pixel_count"],
            "eye": appearance.meta["eye_pixel_count"],
        },
        debug_paths=debug_paths or {},
    )

    return profile


def save_profile(profile: IdentityProfile, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)

    # Save embedding
    np.save(os.path.join(output_dir, "embedding.npy"), profile.embedding)

    # Save crops
    cv2.imwrite(os.path.join(output_dir, "crops", "aligned_face.png"), profile.aligned_face)
    cv2.imwrite(os.path.join(output_dir, "crops", "expanded_face.png"), profile.expanded_face)
    cv2.imwrite(os.path.join(output_dir, "crops", "original_face.png"), profile.original_face)

    # Save masks
    mask_dict = {
        "face.png": profile.mask_face,
        "skin.png": profile.mask_skin,
        "hair.png": profile.mask_hair,
        "eye.png": profile.mask_eye,
        "mouth.png": profile.mask_mouth,
    }

    for name, mask in mask_dict.items():
        cv2.imwrite(os.path.join(output_dir, "masks", name), mask)

    # Save metadata JSON
    json_data = {
        "embedding_norm": profile.embedding_norm,
        "appearance": {
            "hair_hex": profile.hair_color_hex,
            "hair_shade": profile.hair_shade,
            "skin_rgb": profile.skin_color_rgb,
            "skin_tone": profile.skin_tone,
            "eye_rgb": profile.eye_color_rgb,
            "eye_color": profile.eye_color_name,
        },
        "pixel_stats": profile.pixel_stats,
        "roll_angle": profile.roll_angle,
        "debug_paths": profile.debug_paths,
        "shape_info": {
            "aligned_face": profile.aligned_face.shape,
            "expanded_face": profile.expanded_face.shape,
            "original_face": profile.original_face.shape,
        }
    }

    with open(os.path.join(output_dir, "identity_profile.json"), "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"✔ Identity profile saved to: {output_dir}")