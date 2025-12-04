"""
SDXL Face Inpainting with IP-Adapter for identity preservation
"""

import torch
import numpy as np
from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL
from PIL import Image
import cv2


def load_image_for_sdxl(img):
    """Convert BGR NumPy to PIL RGB."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    raise TypeError("Unsupported image type for SDXL.")


class SDXLFaceInpainter:

    def __init__(self,
                 device="cuda",
                 model_name="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"):
        """
        SDXL inpainting with IP-Adapter Plus Face for identity preservation.
        """
        self.device = device

        # Load SDXL inpainting model
        print("\n=== Loading SDXL Inpainting ===")
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
        print("✔ SDXL Inpainting loaded")

        # Load IP-Adapter Plus Face (publicly available, no auth needed)
        print("\n=== Loading IP-Adapter Plus Face ===")
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors"
        )
        self.pipe.set_ip_adapter_scale(0.6)
        print("✔ IP-Adapter Plus Face loaded")

        # High-quality VAE
        print("\n=== Loading VAE ===")
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        ).to(device)
        self.pipe.vae = self.vae
        print("✔ VAE loaded")

        print("\n=== SDXL Face Inpainter Ready ===")

    def prepare_mask(self, mask):
        """Convert mask to PIL RGB."""
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(mask)
        return mask

    def inpaint(self,
                template_img,
                mask,
                profile,
                prompt="",
                negative_prompt="",
                strength=0.85,
                steps=30):
        """
        Run SDXL inpainting with IP-Adapter face guidance.
        
        Args:
            template_img: BGR numpy array or PIL Image
            mask: Binary mask (255 = inpaint area)
            profile: Identity profile with face_crop
            prompt: Text prompt
            negative_prompt: Negative prompt
            strength: Inpainting strength (0.0-1.0)
            steps: Number of inference steps
        """
        # Prepare images
        image = load_image_for_sdxl(template_img)
        mask_image = self.prepare_mask(mask)
        
        # Get face image for IP-Adapter
        face_img = profile.face_crop  # BGR numpy
        face_pil = load_image_for_sdxl(face_img)
        
        # Resize to expected size
        image = image.resize((1024, 1024))
        mask_image = mask_image.resize((1024, 1024))
        face_pil = face_pil.resize((224, 224))

        # Run inpainting with IP-Adapter
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            ip_adapter_image=face_pil,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=7.5,
        ).images[0]

        # Resize back to original size
        orig_h, orig_w = template_img.shape[:2] if isinstance(template_img, np.ndarray) else template_img.size[::-1]
        result = result.resize((orig_w, orig_h))
        
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
