"""
ArcFace Embedding using InsightFace App interface
"""

import cv2
import numpy as np
from dataclasses import dataclass
from insightface.app import FaceAnalysis


# ----------------------------
# DATA STRUCTURE
# ----------------------------

@dataclass
class FaceEmbeddingResult:
    embedding: np.ndarray   # (512,)
    norm: float
    raw_face: np.ndarray


# ----------------------------
# MAIN CLASS
# ----------------------------

class FaceEmbedder:
    def __init__(self, model_name="buffalo_l", ctx_id=-1):
        """
        Modern InsightFace way:
        We load the same model pack as detection (buffalo_l)
        and use the built-in face.embedding vector.
        """
        self.app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=ctx_id)

    # ------------------------------------------------------
    def compute_embedding(self, aligned_face_bgr) -> FaceEmbeddingResult:
        """
        Compute embedding from a 112×112 aligned face crop.
        """

        if aligned_face_bgr is None:
            raise ValueError("aligned_face_bgr is None")

        # InsightFace expects RGB
        face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)

        # Must be exact 112×112
        if face_rgb.shape[:2] != (112, 112):
            face_rgb = cv2.resize(face_rgb, (112, 112))

        # Run embedding
        faces = self.app.get(face_rgb)

        if len(faces) == 0:
            raise RuntimeError("InsightFace returned no embedding for the aligned face.")

        emb = faces[0].embedding  # (512,)

        norm = float(np.linalg.norm(emb))
        emb_normed = (emb / norm).astype(np.float32)

        return FaceEmbeddingResult(
            embedding=emb_normed,
            norm=norm,
            raw_face=aligned_face_bgr
        )

    # ------------------------------------------------------
    def compare(self, emb1, emb2):
        """Cosine similarity"""
        return float(np.dot(emb1, emb2))
