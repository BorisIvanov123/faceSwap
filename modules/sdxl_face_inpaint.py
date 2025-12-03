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
                 model_name="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                 ipadapter_repo="h94/IP-Adapter-FaceID",
                 controlnet_lineart="diffusers/controlnet-sdxl-1.0-lineart",
                 controlnet_depth="diffusers/controlnet-sdxl-1.0-depth"):
        """
        Initialize the full SDXL inpainting stack with identity + structure preserving.
        """

        self.device = device

        print("\n=== Loading SDXL Inpainting ===")
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)

        print("✔ SDXL Inpainting loaded")

        print("\n=== Loading IP-Adapter FaceID ===")
        self.pipe.load_ip_adapter(ipadapter_repo, subfolder="sdxl_faceid")
        print("✔ IP-Adapter FaceID loaded")

        print("\n=== Loading ControlNet-LineArt & Depth ===")
        self.control_lineart = ControlNetModel.from_pretrained(
            controlnet_lineart,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)

        self.control_depth = ControlNetModel.from_pretrained(
            controlnet_depth,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)

        print("✔ ControlNet models loaded")

        # Attach controlnets
        self.pipe.controlnet = [self.control_lineart, self.control_depth]

        # Autoencoder for higher-quality details (optional)
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        ).to(device)
        self.pipe.vae = self.vae

        print("\n=== SDXL Face Inpainter Ready ===")


    # ---------------------------------------------------------------
    # Utility: preprocessing masks & images
    # ---------------------------------------------------------------
    def prepare_mask(self, mask):
        """
        SDXL expects a white mask on the inpaint area.
        mask: uint8 0/255 array
        Returns a PIL image.
        """
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(mask)
        return mask


    def preprocessing(self, template_img, mask):
        pil_image = load_image_for_sdxl(template_img)
        pil_mask = self.prepare_mask(mask)

        # ControlNet preprocessors
        img_np = np.array(pil_image)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 80, 120)          # lineart edges
        depth = img_gray                              # placeholder depth (use real if available)

        control_lineart = Image.fromarray(edges)
        control_depth = Image.fromarray(depth)

        return pil_image, pil_mask, control_lineart, control_depth


    # ---------------------------------------------------------------
    # Main function: inpaint with identity
    # ---------------------------------------------------------------
    def inpaint(self, template_img, mask, profile, prompt="", negative_prompt="", strength=0.22, steps=25):

        # preprocess inputs
        image, inpaint_mask, lineart_cn, depth_cn = self.preprocessing(template_img, mask)

        # Identity conditioning from ArcFace embedding
        identity_emb = torch.tensor(profile.embedding, dtype=torch.float16).unsqueeze(0).to(self.device)

        # SDXL inference
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=inpaint_mask,
            controlnet_conditioning_scale=[0.8, 0.6],       # lineart, depth
            controlnet_hint=[lineart_cn, depth_cn],
            ip_adapter_faceid_emb=identity_emb,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=5.0,
        )

        final = result.images[0]
        return cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)

