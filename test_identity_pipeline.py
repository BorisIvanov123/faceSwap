"""
test_identity_pipeline.py
--------------------------
Runs the full identity processing pipeline:

  1. Face detection
  2. Landmark extraction
  3. ArcFace embedding
  4. Face parsing (BiSeNet)
  5. Appearance extraction

Outputs:
  - diagnostic prints
  - aligned face
  - original face crop
  - segmentation overlay
"""

import os
import cv2

# Correct module imports
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
detector = FaceDetector(ctx_id=-1)     # CPU for now; switch to 0 for GPU
det = detector.detect_faces(img)

if det is None:
    raise RuntimeError("❌ No face detected in the image!")

print("✔ Face detected")
print("Score:", det.detection_score)

# save crops
cv2.imwrite(f"{OUT_DIR}/aligned_face.png", det.aligned_face)
cv2.imwrite(f"{OUT_DIR}/original_face.png", det.original_face)
print("Saved face crops.")


# -------------------------------------------------
# 2. LANDMARK PROCESSING
# -------------------------------------------------

print("\n=== Landmark Processing ===")
lm_proc = FaceLandmarkProcessor()
lm = lm_proc.process(det)

print("Landmarks shape:", lm.landmarks.shape)
print("Roll angle:", lm.roll_angle)


# -------------------------------------------------
# 3. ARC-FACE EMBEDDING
# -------------------------------------------------

print("\n=== ArcFace Embedding ===")
embedder = FaceEmbedder(ctx_id=-1)
emb = embedder.compute_embedding(det.aligned_face)

print("Embedding shape:", emb.embedding.shape)
print("Embedding norm:", emb.norm)
print("Embedding sample:", emb.embedding[:5])


# -------------------------------------------------
# 4. FACE PARSING (BiSeNet)
# -------------------------------------------------

print("\n=== Face Parsing ===")
parser = FaceParser(device="cpu")   # switch to "cuda" if your BiSeNet weights support GPU
parse = parser.parse(det.original_face)

print("Segmentation map shape:", parse.seg_map.shape)

overlay = parser.visualize_masks(parse)
cv2.imwrite(f"{OUT_DIR}/parsing_overlay.png", overlay)
print("Saved parsing overlay.")


# -------------------------------------------------
# 5. APPEARANCE EXTRACTION
# -------------------------------------------------

print("\n=== Appearance Extraction ===")
appearance = AppearanceExtractor().extract(parse)

print("\n==== FINAL APPEARANCE ====")
print("Hair:", appearance.hair_color_hex, "| shade:", appearance.hair_shade)
print("Skin tone:", appearance.skin_tone)
print("Eye color:", appearance.eye_color_name)
print("=================================\n")

print("All stages successful! 🎉")
