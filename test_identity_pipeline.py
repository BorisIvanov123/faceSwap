"""
test_identity_pipeline.py
--------------------------
Runs the full identity processing pipeline:

  1. Face detection
  2. Landmarks
  3. Embedding (ArcFace)
  4. Parsing (BiSeNet)
  5. Appearance extraction

Outputs:
  - prints diagnostic info
  - saves aligned face
  - saves segmentation visualization
"""

import os
import cv2

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

# 1. FACE DETECTION
print("\n=== Face Detection ===")
detector = FaceDetector(ctx_id=-1)   # CPU for now; change to 0 for GPU
det = detector.detect_faces(img)

if det is None:
    raise RuntimeError("❌ No face detected in the image!")
else:
    print("✔ Face detected with score:", det.detection_score)

cv2.imwrite(f"{OUT_DIR}/aligned_face.png", det.aligned_face)
cv2.imwrite(f"{OUT_DIR}/original_crop.png", det.original_face)

print("Saved aligned and original face crops.")


# 2. LANDMARKS
print("\n=== Landmark Processing ===")
lm_proc = FaceLandmarkProcessor()
lm_result = lm_proc.process(det)

print("Landmarks:", lm_result.landmarks.shape)
print("Head roll angle:", lm_result.roll_angle)


# 3. EMBEDDING
print("\n=== ArcFace Embedding ===")
embedder = FaceEmbedder(ctx_id=-1)
emb = embedder.compute_embedding(det.aligned_face)

print("Embedding shape:", emb.embedding.shape)
print("Embedding norm:", emb.norm)
print("Embedding sample:", emb.embedding[:5])


# 4. PARSING
print("\n=== Face Parsing (BiSeNet) ===")
parser = FaceParser(device="cpu")  # change to "cuda" if GPU ready
parse = parser.parse(det.original_face)

print("Seg map shape:", parse.seg_map.shape)

# save overlay visualization
overlay = parser.visualize_masks(parse)
cv2.imwrite(f"{OUT_DIR}/parsing_overlay.png", overlay)
print("Saved parsing overlay.")


# 5. APPEARANCE EXTRACTION
print("\n=== Appearance Extraction ===")
extractor = AppearanceExtractor()
appearance = extractor.extract(parse)

print("\n==== FINAL APPEARANCE ====")
print("Hair:", appearance.hair_color_hex, "| shade:", appearance.hair_shade)
print("Skin tone:", appearance.skin_tone)
print("Eye color:", appearance.eye_color_name)
print("=================================")

print("\nAll stages successful! 🎉")
