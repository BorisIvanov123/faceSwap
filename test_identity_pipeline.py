"""
test_identity_pipeline.py - Full identity pipeline with SegFace parsing
"""

import os
import cv2
import numpy as np

from modules.face_detection import FaceDetector, load_image
from modules.face_landmarks import FaceLandmarkProcessor
from modules.face_embeddings import FaceEmbedder
from modules.face_parsing_segface import FaceParserSegFace, SEGFACE_LABELS
from modules.appearance_extraction import AppearanceExtractor
from modules.identity_profile import build_identity_profile, save_profile
from modules.sdxl_face_inpaint import SDXLFaceInpainter
from modules.prompt_generator import generate_prompt, generate_negative


IMG_PATH = "photos/faces/lape.jpg"
OUT_DIR = "debug_output"
TEMPLATE_PAGE = "photos/templates/1.png"
FACE_REGIONS_JSON = "templates/face_regions.json"


os.makedirs(OUT_DIR, exist_ok=True)

print("\n=== Loading image ===")
img = load_image(IMG_PATH)
print(f"Loaded: {IMG_PATH}, shape={img.shape}")


# 1. FACE DETECTION
print("\n=== Face Detection ===")
detector = FaceDetector(ctx_id=0)
det = detector.detect_faces(img)

if det is None:
    raise RuntimeError("❌ No face detected!")

print(f"✔ Face detected (score: {det.detection_score:.3f})")
print(f"  BBox: {det.bbox}")
print(f"  Original crop: {det.original_face.shape}")
print(f"  Expanded crop: {det.expanded_face.shape}")

cv2.imwrite(f"{OUT_DIR}/1_aligned_face.png", det.aligned_face)
cv2.imwrite(f"{OUT_DIR}/1_original_face.png", det.original_face)
cv2.imwrite(f"{OUT_DIR}/1_expanded_face.png", det.expanded_face)
print("  Saved: aligned, original, expanded crops")


# 2. LANDMARKS
print("\n=== Landmark Processing ===")
lm_proc = FaceLandmarkProcessor()
lm = lm_proc.process(det)
print(f"  Landmarks: {lm.landmarks.shape}, Roll: {lm.roll_angle:.1f}°")


# 3. EMBEDDINGS
print("\n=== ArcFace Embedding ===")
embedder = FaceEmbedder(ctx_id=0)
emb = embedder.compute_embedding(det.aligned_face)
print(f"  Shape: {emb.embedding.shape}, Norm: {emb.norm:.2f}")


# 4. FACE PARSING (SegFace)
print("\n=== Face Parsing (SegFace) ===")
parser = FaceParserSegFace(
    model_path="SegFace/weights/mobilenet_celeba_512/model_299.pt",
    backbone="mobilenet",
    input_resolution=512
)
parse = parser.parse(det.expanded_face)

if parse is None:
    raise RuntimeError("❌ Parsing failed!")

print(f"  Seg map: {parse.seg_map.shape}")
print(f"  Unique labels: {np.unique(parse.seg_map)}")

# Print detailed label statistics
print("\n  Label breakdown:")
for label in sorted(np.unique(parse.seg_map)):
    count = (parse.seg_map == label).sum()
    name = SEGFACE_LABELS.get(label, '?')
    print(f"    {label:2d} ({name:10s}): {count:,} pixels")

# Save visualizations
overlay = parser.visualize_masks(parse)
cv2.imwrite(f"{OUT_DIR}/4_parsing_overlay.png", overlay)

all_labels = parser.visualize_all_labels(parse)
cv2.imwrite(f"{OUT_DIR}/4_all_labels.png", all_labels)

masks_info = [
    ('hair', parse.hair_mask),
    ('skin', parse.skin_mask),
    ('eye', parse.eye_mask),
    ('mouth', parse.mouth_mask),
]

print("\n  Mask pixel counts:")
for name, mask in masks_info:
    pixel_count = (mask > 0).sum()
    print(f"    {name}: {pixel_count:,} pixels")
    cv2.imwrite(f"{OUT_DIR}/4_mask_{name}.png", mask)
    
    masked_region = cv2.cvtColor(parse.processed_img, cv2.COLOR_RGB2BGR).copy()
    masked_region[mask == 0] = 0
    cv2.imwrite(f"{OUT_DIR}/4_masked_{name}.png", masked_region)


# 5. APPEARANCE
print("\n=== Appearance Extraction ===")
extractor = AppearanceExtractor()
appearance = extractor.extract(parse)

print(f"\n  Raw RGB:")
print(f"    Hair: {appearance.hair_color_rgb}")
print(f"    Skin: {appearance.skin_color_rgb}")
print(f"    Eyes: {appearance.eye_color_rgb}")

print("\n" + "="*45)
print("         FINAL APPEARANCE RESULTS")
print("="*45)
print(f"  Hair:  {appearance.hair_color_hex} → {appearance.hair_shade}")
print(f"  Skin:  RGB{appearance.skin_color_rgb} → {appearance.skin_tone}")
print(f"  Eyes:  RGB{appearance.eye_color_rgb} → {appearance.eye_color_name}")
print("="*45)

print(f"\n✅ All stages successful!")
print(f"   Check {OUT_DIR}/4_all_labels.png for full segmentation")
print(f"   Hair=PURPLE, Skin=GREEN, Eyes=BLUE, Mouth=YELLOW")

# 6. BUILD IDENTITY PROFILE
print("\n=== Building Identity Profile ===")
profile = build_identity_profile(
    det=det,
    lm=lm,
    emb=emb,
    parse=parse,
    appearance=appearance,
    debug_paths={
        "aligned_face": f"{OUT_DIR}/1_aligned_face.png",
        "expanded_face": f"{OUT_DIR}/1_expanded_face.png",
        "parsing_overlay": f"{OUT_DIR}/4_parsing_overlay.png",
        "all_labels": f"{OUT_DIR}/4_all_labels.png",
    }
)

print("✔ Identity profile constructed!")

IDENTITY_OUT = os.path.join(OUT_DIR, "identity_profile")
save_profile(profile, IDENTITY_OUT)

print("✔ Identity profile saved for debugging!")

# 6 SDXL FACE INPAINTING WITH IDENTITY


print("\n=== Loading Template Page ===")
page_img = load_image(TEMPLATE_PAGE)
print(f"Loaded template page: {TEMPLATE_PAGE}, shape={page_img.shape}")

# Load the annotated face region for this page
import json
with open(FACE_REGIONS_JSON, "r") as f:
    FACE_REGIONS = json.load(f)

page_name = os.path.basename(TEMPLATE_PAGE)
if page_name not in FACE_REGIONS:
    raise RuntimeError(f"❌ No face region found for {page_name} in {FACE_REGIONS_JSON}")

x1, y1, x2, y2 = FACE_REGIONS[page_name]
print(f"Face region for inpainting: {x1, y1, x2, y2}")

# Create the inpainting mask
mask = np.zeros(page_img.shape[:2], dtype=np.uint8)
mask[y1:y2, x1:x2] = 255
cv2.imwrite(f"{OUT_DIR}/7_inpaint_mask.png", mask)
print("✔ Mask saved for debugging")

# Generate prompt based on profile appearance
prompt = generate_prompt(profile, page_context="children's Christmas storybook scene")
negative = generate_negative()

print("\n=== Running SDXL Face Inpainting ===")

inpainter = SDXLFaceInpainter(device="cuda")

result = inpainter.inpaint(
    template_img=page_img,
    mask=mask,
    profile=profile,
    prompt=prompt,
    negative_prompt=negative,
    strength=0.95,
    steps=30
)

# Save output
out_path = f"{OUT_DIR}/8_sdxl_result.png"
cv2.imwrite(out_path, result)
print(f"✔ SDXL inpainting complete!")
print(f"Result saved to {out_path}")
