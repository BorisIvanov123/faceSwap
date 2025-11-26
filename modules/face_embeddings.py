"""
face_embedding.py
-------------------
Compute identity embeddings with ArcFace (InsightFace).

Consumes:
  - aligned face from face_detection.FaceDetectionResult
    (aligned_face = 112x112 ArcFace-ready image)

Produces:
  - 512-d identity embedding (L2-normalized)
  - Used by IP-Adapter FaceID, identity consistency checks,
    and throughout the entire pipeline.

Notes:
  - MUST be stable across slightly different crops/poses
  - MUST be normalized for cosine similarity
"""

import numpy as np
import cv2
from insightface.model_zoo import get_model
from dataclasses import dataclass


# ----------------------------
# DATA STRUCTURE
# ----------------------------

@dataclass
class FaceEmbeddingResult:
    embedding: np.ndarray   # shape (512,)
    norm: float             # should be ~1.0
    raw_face: np.ndarray    # original aligned face (112x112)


# ----------------------------
# MAIN CLASS
# ----------------------------

class FaceEmbedder:
    def __init__(self, model_path=None, ctx_id=-1):
        """
        Initialize ArcFace model.

        Params:
          model_path (str): custom model file, else loads default ArcFace
          ctx_id: -1 = CPU, 0 = GPU 0
        """
        if model_path:
            self.model = get_model(model_path, root='.')
        else:
            # Default ArcFace 512d model
            self.model = get_model('arcface_r100_v1', root='.')

        self.model.prepare(ctx_id=ctx_id)

    # ------------------------------------------------------
    def compute_embedding(self, aligned_face_bgr) -> FaceEmbeddingResult:
        """
        Compute a single face embedding.
        Input must be a 112x112 aligned face.
        """
        if aligned_face_bgr is None:
            raise ValueError("aligned_face_bgr is None")

        # InsightFace ArcFace model expects RGB input
        face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)

        # Input must be exactly 112x112
        if face_rgb.shape[:2] != (112, 112):
            face_rgb = cv2.resize(face_rgb, (112, 112))

        # Compute embedding
        emb = self.model.get_embedding(face_rgb).flatten()  # shape (512,)

        # Normalize for cosine similarity
        norm = np.linalg.norm(emb)
        if norm == 0:
            raise ValueError("ArcFace embedding norm is zero. Bad input?")

        emb_normalized = emb / norm

        return FaceEmbeddingResult(
            embedding=emb_normalized.astype(np.float32),
            norm=float(norm),
            raw_face=aligned_face_bgr
        )

    # ------------------------------------------------------
    def compare(self, emb1, emb2):
        """
        Cosine similarity between two embeddings.
        """
        return float(np.dot(emb1, emb2))


# ----------------------------
# DEMO / DEBUG
# ----------------------------

if __name__ == "__main__":
    from face_detection import FaceDetector, load_image

    TEST_IMG = "../photos/faces/animated_test.jpeg"

    img = load_image(TEST_IMG)

    # Detect
    detector = FaceDetector(ctx_id=-1)
    det = detector.detect_faces(img)

    if det is None:
        print("❌ No face detected, aborting")
        exit()

    # Embed
    embedder = FaceEmbedder(ctx_id=-1)
    result = embedder.compute_embedding(det.aligned_face)

    print("✔ Embedding computed!")
    print("Shape:", result.embedding.shape)
    print("Norm:", result.norm)
    print("Sample values:", result.embedding[:5])
