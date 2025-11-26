import time
import cv2
from insightface.app import FaceAnalysis

# -------- CONFIG --------
IMG_PATH = "../photos/faces/animated_test.jpeg"  # <-- put your test file here
OUT_PATH = "../photos/output_faces/detected_output_animated.jpg"
DET_SIZE = (640, 640)          # detection resolution (we can tune this later)

# -------- INIT --------
print("\nInitializing InsightFace...")
start_init = time.time()

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=DET_SIZE)   # CPU-only on Mac

end_init = time.time()
print(f"Initialization time: {end_init - start_init:.2f} seconds\n")

# -------- LOAD IMAGE --------
print(f"Loading: {IMG_PATH}")
img = cv2.imread(IMG_PATH)

if img is None:
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")

# -------- RUN DETECTION --------
print("\nRunning face detection...")
start_det = time.time()

faces = app.get(img)

end_det = time.time()
elapsed = end_det - start_det

print(f"Detection time: {elapsed:.4f} seconds")
print(f"Faces detected: {len(faces)}")

# -------- PRINT DIAGNOSTICS --------
for i, face in enumerate(faces):
    print(f"\nFace #{i+1}")
    print("  Bounding Box:", face.bbox)
    print("  Detection Score:", getattr(face, "det_score", "N/A"))
    print("  Landmark Count:", len(face.landmark_2d_106))

# -------- SAVE VISUALIZATION --------
if len(faces) > 0:
    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imwrite(OUT_PATH, img)
    print(f"\nOutput saved to: {OUT_PATH}")
else:
    print("\nNo faces detected. No output saved.")
