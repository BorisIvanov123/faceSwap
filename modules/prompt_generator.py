def generate_prompt(profile, page_context="storybook illustration"):
    hair = profile.hair_shade.replace("_", " ")
    skin = profile.skin_tone.replace("_", " ")
    eyes = profile.eye_color_name.replace("_", " ")

    prompt = (
        f"A young child with {hair} hair, {skin} skin, and {eyes} eyes, "
        f"in a soft children's book illustration style. "
        f"{page_context}. Warm lighting, clean lines, gentle colors, consistent identity."
    )

    return prompt


def generate_negative():
    negative = (
        "wrong identity, wrong face, photorealistic, adult, distorted facial features, "
        "mismatched skin tone, mismatched hair color, extra eyes, extra mouth, "
        "glitch, deformed face, bad anatomy, blurry face, broken lineart, "
        "high realism, uncanny valley"
    )
    return negative
