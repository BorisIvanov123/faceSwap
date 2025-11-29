"""
test_identity_pipeline.py
--------------------------
Runs the full identity processing pipeline:

  1. Face detection
  2. Landmark extraction
  3. ArcFace embedding
  4. Face parsing (BiSeNet ONNX)
  5. Appearance extraction

Outputs:
  - diagnostic prints
  - aligned face
  - original face crop
  - segmentation overlay
  - individual mask debug images
"""

import os
import cv2
import numpy as np

from modules.face_detection import FaceDetector, load_image
from modules.face_landmarks import FaceLandmarkProcessor
from modules.face_embeddings import FaceEmbedder
from modules.face_parsing import FaceParser
from modules.appearance_extraction import AppearanceExtractor


# ----------------------------
# CONFIG
# ----------------------------

IMG_PATH = "photos/faces/test.jpg"
OUT_DIR = "debug_output"

os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------
# RUN PIPELINE
# ----------------------------

print("\n=== Loading image ===")
img = load_image(IMG_PATH)
print(f"Loaded: {IMG_PATH}, shape={img.shape}")


# -------------------------------------------------
# 1. FACE DETECTION
# -------------------------------------------------

print("\n=== Face Detection ===")
detector = FaceDetector(ctx_id=0)
det = detector.detect_faces(img)

if det is None:
    raise RuntimeError("❌ No face detected in the image!")

print(f"✔ Face detected (score: {det.detection_score:.3f})")
print(f"  BBox: {det.bbox}")

cv2.imwrite(f"{OUT_DIR}/1_aligned_face.png", det.aligned_face)
cv2.imwrite(f"{OUT_DIR}/1_original_face.png", det.original_face)
print("  Saved: 1_aligned_face.png, 1_original_face.png")


# -------------------------------------------------
# 2. LANDMARK PROCESSING
# -------------------------------------------------

print("\n=== Landmark Processing ===")
lm_proc = FaceLandmarkProcessor()
lm = lm_proc.process(det)

print(f"  Landmarks shape: {lm.landmarks.shape}")
print(f"  Roll angle: {lm.roll_angle:.1f}°")


# -------------------------------------------------
# 3. ARC-FACE EMBEDDING
# -------------------------------------------------

print("\n=== ArcFace Embedding ===")
embedder = FaceEmbedder(ctx_id=0)
emb = embedder.compute_embedding(det.aligned_face)

print(f"  Embedding shape: {emb.embedding.shape}")
print(f"  Embedding norm: {emb.norm:.2f}")
print(f"  Sample: {emb.embedding[:5]}")


# -------------------------------------------------
# 4. FACE PARSING (BiSeNet ONNX)
# -------------------------------------------------

print("\n=== Face Parsing (BiSeNet) ===")
parser = FaceParser(model_path="weights/resnet18.onnx", ctx_id=0)
parse = parser.parse(det.original_face)

if parse is None:
    raise RuntimeError("❌ Parsing failed — segmentation returned None")

print(f"  Segmentation map shape: {parse.seg_map.shape}")
print(f"  Unique labels found: {np.unique(parse.seg_map)}")

# Save parsing overlay
overlay = parser.visualize_masks(parse)
cv2.imwrite(f"{OUT_DIR}/4_parsing_overlay.png", overlay)

# Save individual masks and masked regions for debugging
masks_info = [
    ('hair', parse.hair_mask),
    ('skin', parse.skin_mask),
    ('eye', parse.eye_mask),
    ('mouth', parse.mouth_mask),
    ('face', parse.face_mask),
]

print("\n  Mask pixel counts:")
for name, mask in masks_info:
    pixel_count = (mask > 0).sum()
    print(f"    {name}: {pixel_count:,} pixels")

    # Save raw mask
    cv2.imwrite(f"{OUT_DIR}/4_mask_{name}.png", mask)

    # Save masked region (shows actual pixels being sampled)
    masked_region = det.original_face.copy()
    masked_region[mask == 0] = 0
    cv2.imwrite(f"{OUT_DIR}/4_masked_{name}.png", masked_region)

print(f"\n  Saved all masks to {OUT_DIR}/")


# -------------------------------------------------
# 5. APPEARANCE EXTRACTION
# -------------------------------------------------

print("\n=== Appearance Extraction ===")
extractor = AppearanceExtractor()
appearance = extractor.extract(parse)

print(f"\n  Raw RGB values extracted:")
print(f"    Hair RGB: {appearance.hair_color_rgb}")
print(f"    Skin RGB: {appearance.skin_color_rgb}")
print(f"    Eye RGB:  {appearance.eye_color_rgb}")

print("\n" + "="*45)
print("         FINAL APPEARANCE RESULTS")
print("="*45)
print(f"  Hair:  {appearance.hair_color_hex} → {appearance.hair_shade}")
print(f"  Skin:  RGB{appearance.skin_color_rgb} → {appearance.skin_tone}")
print(f"  Eyes:  RGB{appearance.eye_color_rgb} → {appearance.eye_color_name}")
print("="*45)

print(f"\n✅ All stages successful!")
print(f"   Debug outputs saved to: {OUT_DIR}/")
print(f"   Check 4_masked_hair.png to verify hair mask quality")