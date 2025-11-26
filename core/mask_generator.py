"""
Mask Generator - Creates masks for face replacement in templates
OPTIMIZED: Smaller mask for face + hair only, excludes body/clothes
"""

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw


class MaskGenerator:
    """Generate masks for face replacement"""

    def __init__(self):
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=-1, det_size=(640, 640))

    def generate_ellipse_mask(self, template_path, mask_path, expansion_factor=1.2):
        """
        Generate an elliptical mask around detected face.

        OPTIMIZED: Smaller mask (1.2x instead of 1.8x) to cover ONLY face + hair,
        not body/clothes/shoulders.
        """

        # Load and detect
        img = cv2.imread(template_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load {template_path}")

        faces = self.face_app.get(img)
        if len(faces) == 0:
            raise Exception("No face detected in template!")

        face = faces[0]
        h, w = img.shape[:2]

        # Get face bbox
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Calculate center and dimensions
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        fw = x2 - x1
        fh = y2 - y1

        # OPTIMIZED: Smaller expansion for face + hair ONLY
        # Horizontal: 1.2x (just enough for hair width)
        radius_x = int(fw * expansion_factor / 2)

        # Vertical asymmetric: more on top for hair, less below for chin
        radius_y_top = int(fh * expansion_factor * 0.5)      # Reduced from 0.7
        radius_y_bottom = int(fh * expansion_factor * 0.25)  # Reduced from 0.4

        # Create mask
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)

        # OPTIMIZED: Adjust center MORE upward to focus on face/hair
        cy_adjusted = cy - int(fh * 0.15)  # Increased from 0.1

        ellipse_bbox = [
            cx - radius_x,
            cy_adjusted - radius_y_top,
            cx + radius_x,
            cy_adjusted + radius_y_bottom
        ]

        draw.ellipse(ellipse_bbox, fill=255)

        # OPTIMIZED: Heavier feathering for smoother blending
        mask = np.array(mask)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)  # Increased from (21, 21)
        mask = Image.fromarray(mask)

        # Save
        mask.save(mask_path)
        print(f"✅ Precise face+hair mask saved to: {mask_path}")

        return mask

    def generate_smart_mask(self, template_path, mask_path, method="ellipse"):
        """Smart mask generation"""
        if method == "ellipse":
            return self.generate_ellipse_mask(template_path, mask_path)
        else:
            raise ValueError(f"Unknown method: {method}")

    def visualize_mask(self, template_path, mask_path, output_path):
        """Overlay mask on template to visualize what will be replaced"""
        template = cv2.imread(template_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Create red overlay where mask is active
        overlay = template.copy()
        overlay[mask > 128] = [0, 0, 255]  # Red = will be replaced

        # Blend
        result = cv2.addWeighted(template, 0.6, overlay, 0.4, 0)

        cv2.imwrite(output_path, result)
        print(f"✅ Visualization saved to: {output_path}")
        print(f"   Red area = what will be replaced")