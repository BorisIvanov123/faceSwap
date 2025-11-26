"""
Page Generator - Generates personalized storybook pages
OPTIMIZED: No resize, better parameters for face+hair only replacement
"""

import torch
import numpy as np
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image
import cv2
from insightface.app import FaceAnalysis

# Use CPU to avoid memory issues (change to "cuda" on NVIDIA)
DEVICE = "cpu"
DTYPE = torch.float32

print(f"Using device: {DEVICE}")


class SimplifiedBookGenerator:
    """Generate personalized pages using ControlNet"""

    def __init__(self):
        print("Loading models (this may take a few minutes first time)...")

        # Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            torch_dtype=DTYPE
        ).to(DEVICE)

        # Load SD pipeline
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=self.controlnet,
            torch_dtype=DTYPE,
            safety_checker=None
        ).to(DEVICE)

        # Memory optimizations
        self.pipe.enable_attention_slicing(1)
        self.pipe.enable_vae_slicing()

        # InsightFace
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=-1, det_size=(640, 640))

        print("✅ Models loaded\n")

    def analyze_child_face(self, image_path):
        """Extract face features for prompting"""
        img = cv2.imread(image_path)
        faces = self.face_app.get(img)

        if len(faces) == 0:
            raise Exception("No face detected")

        face = faces[0]
        age = int(face.age) if hasattr(face, 'age') else 8
        gender = face.gender if hasattr(face, 'gender') else 1
        gender_str = "girl" if gender == 1 else "boy"

        return f"{age} year old {gender_str}", face

    def create_canny_edge(self, image, low_threshold=100, high_threshold=200):
        """Create canny edge map for ControlNet"""
        image_np = np.array(image)
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(image_gray, low_threshold, high_threshold)
        edges = Image.fromarray(edges)
        return edges

    def generate_page(
        self,
        child_photo_path,
        template_path,
        mask_path,
        output_path,
        scene_description=""
    ):
        """Generate a personalized page"""

        print("\n=== Generating Page ===")

        # Analyze face
        print("Analyzing child's face...")
        child_desc, face = self.analyze_child_face(child_photo_path)
        print(f"  Detected: {child_desc}")

        # Load template and mask - NO RESIZE, keep original resolution
        print("Loading template at original resolution...")
        template = load_image(template_path).convert("RGB")
        mask = load_image(mask_path).convert("L")

        # Ensure mask matches template size
        if template.size != mask.size:
            print(f"  Resizing mask to match template: {template.size}")
            mask = mask.resize(template.size, Image.Resampling.LANCZOS)

        print(f"  Resolution: {template.size[0]}x{template.size[1]}")

        # Create edge map
        print("Creating edge map...")
        canny_image = self.create_canny_edge(template)

        # OPTIMIZED PROMPT: Emphasize preservation of everything except face+hair
        prompt = (
            f"face portrait, {child_desc}, realistic child face, natural hair, "
            f"{scene_description}, "
            "photorealistic face, detailed eyes, natural skin texture, "
            "preserve background, preserve scene, preserve composition, "
            "only face and hair changed, same lighting, same style"
        )

        negative_prompt = (
            "ugly, distorted, disfigured, bad anatomy, bad proportions, "
            "extra limbs, blurry, low quality, adult face, old, "
            "deformed face, mutated, extra fingers, text corruption, "
            "watermark, signature"
        )

        print(f"  Prompt: {prompt[:100]}...")
        print("Generating (this will take 3-5 minutes on CPU at full resolution)...")

        # OPTIMIZED PARAMETERS for face+hair replacement only
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=template,
            mask_image=mask,
            control_image=canny_image,
            num_inference_steps=25,                    # Increased from 20 for better quality
            guidance_scale=7.5,                        # Same
            controlnet_conditioning_scale=0.3,         # REDUCED from 0.7 - less preservation of original
            strength=0.95,                             # INCREASED from 0.85 - replace more
        ).images[0]

        # Save at high quality
        result.save(output_path, quality=95)
        print(f"✅ Saved to: {output_path}")

        return result