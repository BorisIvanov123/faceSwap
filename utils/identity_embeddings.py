"""
Extract and save face embeddings for use with generation
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceEmbeddingExtractor:
    def __init__(self):
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=-1, det_size=(640, 640))

    def extract_embedding(self, image_path, output_path):
        """Extract face embedding and save it"""

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load {image_path}")

        # Detect faces
        faces = self.face_app.get(img)
        if len(faces) == 0:
            raise Exception("No faces detected.")

        face = faces[0]

        # Extract embedding (512D vector)
        embedding = np.asarray(face.normed_embedding, dtype=np.float32)

        # Save
        np.save(output_path, embedding)

        print(f"✅ Face embedding extracted and saved to: {output_path}")
        print(f"   Shape: {embedding.shape}")
        print(f"   Norm: {np.linalg.norm(embedding):.3f}")

        # Also return face image crop for IP-Adapter
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Expand bbox for full face
        h, w = img.shape[:2]
        margin = int((x2 - x1) * 0.3)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        face_crop = img[y1:y2, x1:x2]

        return embedding, face_crop


if __name__ == "__main__":
    extractor = FaceEmbeddingExtractor()

    # Extract embedding
    embedding, face_crop = extractor.extract_embedding(
        "photos/faces/test_five.jpg",
        "photos/faces/test_five_embedding.npy"
    )

    # Save face crop too
    cv2.imwrite("photos/faces/test_five_face_crop.jpg", face_crop)
    print("✅ Face crop saved for IP-Adapter")