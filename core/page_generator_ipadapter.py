"""
Page Generator with IP-Adapter FaceID
Uses actual face embeddings for accurate face matching
"""

import torch
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
from PIL import Image
import cv2
from insightface.app import FaceAnalysis

# Import IP-Adapter
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

print(f"Using device: {DEVICE}")


class IPAdapterBookGenerator:
    """Generate pages using IP-Adapter FaceID for accurate face matching"""
    
    def __init__(self):
        print("Loading models with IP-Adapter FaceID...")
        
        # Load base SD pipeline
        base_model = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            base_model,
            torch_dtype=DTYPE,
            safety_checker=None
        )
        
        # Memory optimizations
        if DEVICE == "cpu":
            self.pipe.enable_attention_slicing(1)
            self.pipe.enable_vae_slicing()
        
        # Load IP-Adapter FaceID
        ip_ckpt = "models/ip-adapter-faceid_sd15.bin"
        self.ip_model = IPAdapterFaceID(self.pipe, ip_ckpt, DEVICE)
        
        # InsightFace
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=-1, det_size=(640, 640))
        
        print("✅ IP-Adapter FaceID loaded\n")
    
    def load_face_embedding(self, image_path):
        """Load face and extract embedding"""
        
        img = cv2.imread(image_path)
        faces = self.face_app.get(img)
        
        if len(faces) == 0:
            raise Exception("No face detected")
        
        face = faces[0]
        
        # Get embedding as numpy array
        embedding = face.normed_embedding
        
        # Convert to torch tensor and add batch dimension
        embedding_tensor = torch.from_numpy(embedding).unsqueeze(0).to(DEVICE)
        
        # Crop face for IP-Adapter
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = img.shape[:2]
        margin = int((x2 - x1) * 0.3)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        face_crop = img[y1:y2, x1:x2]
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        face_pil = face_pil.resize((256, 256))
        
        age = int(face.age) if hasattr(face, 'age') else 8
        gender = "girl" if face.gender == 1 else "boy"
        
        return embedding_tensor, face_pil, f"{age} year old {gender}"
    
    def round_to_8(self, value):
        """Round value to nearest multiple of 8"""
        return (value // 8) * 8
    
    def generate_page(
        self,
        child_photo_path,
        template_path,
        mask_path,
        output_path,
        scene_description=""
    ):
        """Generate page with IP-Adapter face matching"""
        
        print("\n=== Generating with IP-Adapter FaceID ===")
        
        # Load face
        print("Extracting face embedding...")
        embedding, face_img, child_desc = self.load_face_embedding(child_photo_path)
        print(f"  Detected: {child_desc}")
        
        # Load template and mask
        print("Loading template...")
        template = load_image(template_path).convert("RGB")
        mask = load_image(mask_path).convert("L")
        
        # Round dimensions to multiples of 8 (required by SD)
        orig_width, orig_height = template.size
        width = self.round_to_8(orig_width)
        height = self.round_to_8(orig_height)
        
        if (width, height) != (orig_width, orig_height):
            print(f"  Adjusting resolution from {orig_width}x{orig_height} to {width}x{height} (divisible by 8)")
            template = template.resize((width, height), Image.Resampling.LANCZOS)
            mask = mask.resize((width, height), Image.Resampling.LANCZOS)
        elif template.size != mask.size:
            mask = mask.resize(template.size, Image.Resampling.LANCZOS)
        
        print(f"  Resolution: {width}x{height}")
        
        # Prompt
        prompt = (
            f"photo of {child_desc}, {scene_description}, "
            "realistic face, natural features, preserve background"
        )
        
        negative_prompt = "ugly, distorted, disfigured, blurry"
        
        print(f"  Prompt: {prompt[:80]}...")
        print("Generating (3-5 minutes on CPU)...")
        
        # Generate with IP-Adapter
        images = self.ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            faceid_embeds=embedding,
            face_image=face_img,
            image=template,
            mask_image=mask,
            num_samples=1,
            width=width,
            height=height,
            num_inference_steps=30,
            seed=42,
        )
        
        result = images[0]
        
        # Resize back to original dimensions if needed
        if (width, height) != (orig_width, orig_height):
            print(f"  Resizing back to original: {orig_width}x{orig_height}")
            result = result.resize((orig_width, orig_height), Image.Resampling.LANCZOS)
        
        result.save(output_path, quality=95)
        print(f"✅ Saved to: {output_path}")
        
        return result
