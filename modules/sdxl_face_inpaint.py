import torch
import numpy as np
from diffusers import StableDiffusionXLInpaintPipeline, ControlNetModel, AutoencoderKL
from PIL import Image
import cv2
import os


def load_image_for_sdxl(img):
    """Convert BGR NumPy to PIL RGB, or pass-through if already PIL."""
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
        Full SDXL inpainting stack:
        - SDXL inpaint model
        - IP-Adapter FaceID for identity preservation
        - ControlNet LineArt + Depth for structure
        - Optional VAE fix for detail
        """

        self.device = device

        # ---------------------------------------------------------------
        # Load SDXL inpainting model
        # ---------------------------------------------------------------
        print("\n=== Loading SDXL Inpainting ===")
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
        print("✔ SDXL Inpainting loaded")

        # ---------------------------------------------------------------
        # Load IP-Adapter FaceID (SDXL version)
        # ---------------------------------------------------------------
        print("\n=== Loading IP-Adapter FaceID (SDXL) ===")
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter-FaceID-SDXL",
            subfolder="models",
            weight_name="ip-adapter-faceid_sdxl.bin"
        )
        print("✔ IP-Adapter FaceID SDXL loaded")

        # ---------------------------------------------------------------
        # LineArt + Depth ControlNet
        # ---------------------------------------------------------------
        print("\n=== Loading ControlNet-LineArt & Depth ===")

        self.control_lineart = ControlNetModel.from_pretrained(
            "diffusers/controlnet-sdxl-1.0-lineart",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)

        self.control_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-sdxl-1.0-depth",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)

        self.pipe.controlnet = [self.control_lineart, self.control_depth]

        print("✔ ControlNet models loaded")

        # ---------------------------------------------------------------
        # High-quality VAE (optional but improves face detail)
        # ---------------------------------------------------------------
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        ).to(device)

        self.pipe.vae = self.vae

        print("\n=== SDXL Face Inpainter Ready ===")

    # ============================================================
    # Mask preparation
    # ============================================================
    def prepare_mask(self, mask):
        """
        SDXL expects an RGB mask where white (255) = inpaint area.
        """
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(mask)
        return mask

    # ============================================================
    # Preprocessing for ControlNet (lineart + depth)
    # ============================================================
    def preprocessing(self, template_img, mask):

        # Convert to PIL
        pil_image = load_image_for_sdxl(template_img)
        pil_mask = self.prepare_mask(mask)

        # ControlNet preprocessors
        img_np = np.array(pil_image)

        # LineArt edges
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 80, 120)

        # Simple depth proxy (you can replace with MiDaS later)
        depth = img_gray

        control_lineart = Image.fromarray(edges)
        control_depth = Image.fromarray(depth)

        return pil_image, pil_mask, control_lineart, control_depth

    # ============================================================
    # Main inpainting function
    # ============================================================
    def inpaint(self,
                template_img,
                mask,
                profile,
                prompt="",
                negative_prompt="",
                strength=0.22,
                steps=25):

        # Prepare images
        image, mask_image, lineart_cn, depth_cn = self.preprocessing(template_img, mask)

        # Identity embedding (ArcFace -> IPAdapter)
        identity_emb = torch.tensor(profile.embedding, dtype=torch.float16).unsqueeze(0).to(self.device)

        # -----------------------------------------------------------
        # Run SDXL Inpaint + IP-Adapter
        # -----------------------------------------------------------
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            controlnet_hint=[lineart_cn, depth_cn],
            controlnet_conditioning_scale=[0.8, 0.6],  # lineart, depth
            ip_adapter_faceid_emb=identity_emb,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=5.0,
        )

        final = result.images[0]
        return cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)
