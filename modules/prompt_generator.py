"""
Prompt generator for SDXL face inpainting
"""

def generate_prompt(profile, page_context="storybook illustration"):
    """Generate a detailed prompt emphasizing the child's appearance."""
    
    hair = profile.hair_shade.replace("_", " ")
    skin = profile.skin_tone.replace("_", " ")
    eyes = profile.eye_color_name.replace("_", " ")
    
    # Map skin tones to more descriptive terms
    skin_description = {
        "very light": "very fair, pale",
        "light": "fair, light",
        "medium": "medium, olive",
        "tan": "tan, warm brown",
        "dark": "dark brown, rich dark"
    }.get(skin, skin)
    
    # Build detailed prompt
    prompt = (
        f"A young African American child with {skin_description} skin, "
        f"{hair} very short hair, and {eyes} eyes, "
        f"happy smiling expression, round face, "
        f"illustrated in a soft Disney Pixar children's book style, "
        f"{page_context}, "
        f"warm lighting, clean lines, gentle colors, "
        f"consistent character design, high quality illustration"
    )
    
    return prompt


def generate_negative():
    """Generate negative prompt to avoid common issues."""
    negative = (
        "white skin, pale skin, caucasian, wrong skin tone, wrong ethnicity, "
        "long hair, straight hair, blonde hair, "
        "photorealistic, adult, old person, "
        "distorted face, deformed, extra limbs, "
        "bad anatomy, blurry, low quality, "
        "mismatched colors, inconsistent style"
    )
    return negative
