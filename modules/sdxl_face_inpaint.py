"""
SDXL Face Inpainting - No resize, preserve original template
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
        self.device = device

        print("\n=== Loading SDXL Inpainting ===")
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
        print("✔ SDXL Inpainting loaded")

        print("\n=== Loading VAE ===")
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        ).to(device)
        self.pipe.vae = self.vae
        print("✔ VAE loaded")

        print("\n=== SDXL Face Inpainter Ready ===")

    def prepare_mask(self, mask):
        """Convert mask to PIL Image."""
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
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
        Run SDXL inpainting at ORIGINAL resolution.
        No resizing - keeps template exactly as is.
        """
        # Convert to PIL
        image = load_image_for_sdxl(template_img)
        mask_image = self.prepare_mask(mask)
        
        print(f"  Image size: {image.size}")
        print(f"  Mask size: {mask_image.size}")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Strength: {strength}, Steps: {steps}")

        # Run inpainting at original size
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=7.5,
        ).images[0]
        
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
