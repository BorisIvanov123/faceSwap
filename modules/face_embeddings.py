"""
face_embedding.py
-------------------
Compute identity embeddings using ArcFace (InsightFace).

Consumes:
  - aligned_face (112x112 RGB/BGR) from FaceDetectionResult

Produces:
  - 512-D L2-normalized embedding
  - Stable identity vector used in all downstream tasks
"""

import numpy as np
import cv2
from insightface.model_zoo import get_model
from dataclasses import dataclass


# ============================
# DATA STRUCTURE
# ============================

@dataclass
class FaceEmbeddingResult:
    embedding: np.ndarray      # (512,) float32
    norm: float                # L2 norm (should be ~1.0)
    raw_face: np.ndarray       # input face crop (112x112 BGR)


# ============================
# EMBEDDING CLASS
# ============================

class FaceEmbedder:
    def __init__(self, model_name="arcface_r100_v1", ctx_id=-1):
        """
        Initialize ArcFace model.

        ctx_id:
          - -1 → CPU
          - 0 → GPU 0
        """
        # Load ArcFace model directly from InsightFace
        self.model = get_model(model_name)
        self.model.prepare(ctx_id=ctx_id)

    # ---------------------------------------------

    def compute_embedding(self, aligned_face_bgr: np.ndarray) -> FaceEmbeddingResult:
        """
        Compute a stable 512-D identity embedding.
        Input: aligned BGR face (112x112)
        """

        if aligned_face_bgr is None:
            raise ValueError("aligned_face_bgr is None")

        # Convert to RGB (ArcFace expects RGB)
        face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)

        # Ensure correct size
        if face_rgb.shape[:2] != (112, 112):
            face_rgb = cv2.resize(face_rgb, (112, 112))

        # Compute embedding
        emb = self.model.get_embedding(face_rgb).flatten()

        # L2-norm
        norm = float(np.linalg.norm(emb))
        if norm == 0:
            raise RuntimeError("Invalid embedding norm (0). Check input.")

        emb = emb / norm

        return FaceEmbeddingResult(
            embedding=emb.astype(np.float32),
            norm=norm,
            raw_face=aligned_face_bgr,
        )

    # ---------------------------------------------

    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Cosine similarity of two identity vectors.
        """
        return float(np.dot(emb1, emb2))


# ============================
# DEBUG / STANDALONE RUN
# ============================

if __name__ == "__main__":
    from face_detection import FaceDetector, load_image

    TEST_IMG = "../photos/faces/test.jpg"

    img = load_image(TEST_IMG)

    # Detect
    detector = FaceDetector(ctx_id=-1)
    det = detector.detect_faces(img)

    if det is None:
        print("❌ No face detected")
        exit()

    # Embed
    embedder = FaceEmbedder(ctx_id=-1)
    result = embedder.compute_embedding(det.aligned_face)

    print("✔ Embedding computed")
    print("Shape:", result.embedding.shape)
    print("Norm:", result.norm)
    print("First 10 values:", result.embedding[:10])
