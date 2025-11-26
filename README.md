cd /workspace/faceSwap
python3.10 -m venv venv
source venv/bin/activate



This means you are now inside your AI environment and can run anything (diffusers, torch, IP-Adapter, ControlNet, etc.)


pip list | grep -E "diffusers|transformers|control|adapter|lora|clip|timm|einops|gdown|safetensors|accelerate"


✅ TOP-LEVEL SYSTEM STRUCTURE

The whole pipeline can be cleanly divided into 5 modules:

Identity Processing Module

Template/Page Module

Generation Module (Diffusion + ControlNet + IP-Adapter)

Book Assembly Module

API / Frontend / Orchestration Module

Now let’s list the scripts inside each.

📦 1. Identity Processing Module

Handles input child photo → embeddings + attributes.

Scripts required:
1. face_detection.py

Loads RetinaFace

Finds face bounding box

Crops + aligns the face

2. face_landmarks.py

Extracts 5–106 landmarks

Used for alignment + QC

3. face_embedding.py

Runs ArcFace

Outputs 512-d identity vector

Normalizes + stores it

4. face_parsing.py

Uses BiSeNet to extract:

hair mask

skin region

eyes

lips

5. appearance_extraction.py

Calculates:

hair color

skin tone

eye color

Converts them to prompt tokens

6. identity_profile.py

Stores everything in a neat JSON:

{
  "embedding": [...],
  "hair_color": "brown",
  "skin_tone": "light",
  "eye_color": "green",
  "encoded_face": "path/to/face.png"
}

📄 2. Template/Page Module

This is for your storybook template pages.

7. page_metadata_loader.py

Loads:

page image

mask (head region)

pose guide (ControlNet map)

prompt template

optional foreground layers

8. mask_generator.py

If needed, creates or resizes masks.

9. pose_map_loader.py

Loads the skeletal/face keypoints map for that page.

10. prompt_builder.py

Creates final prompt:

"{page_prompt}, child with {hair_color} hair, {skin_tone} skin, {eye_color} eyes"

🎨 3. Generation Module (the core)

Handles SDXL, IP-Adapter, ControlNet, Inpainting.

11. sdxl_pipeline_loader.py

Loads SDXL base + refiner

Loads SDXL inpainting model

Loads ControlNet modules

Loads IP-Adapter FaceID

Configures schedulers, samplers

12. inpaint_generator.py

The main function:

generate_page(
   base_page,
   mask,
   pose_map,
   identity_embedding,
   appearance_tokens,
   prompt,
   negative_prompt
)


Outputs the final edited page.

13. identity_control.py

Converts ArcFace embedding → IP-Adapter tokens

Injects into UNet conditioning

14. pose_control.py

Controls OpenPose/Face ControlNet inputs

Ensures pose matches template exactly

15. postprocess_image.py

Sharpen

Color-correct

Texture blending

Ensures result matches template style

📚 4. Book Assembly Module
16. text_personalizer.py

Replaces placeholders {name} in story text

Exports page-text combinations

17. pdf_builder.py

Combines generated pages → PDF

Adds text layers

Creates output file

18. exporter.py

Saves PNGs + PDFs

Organizes output folders

Versioning for each child/book

🌐 5. API / Orchestration Module
19. api_server.py

(Flask/FastAPI)
Endpoints:

/upload-photo

/generate-book

/get-status

/download

20. pipeline_orchestrator.py

This script ties everything together:

Process child photo → identity profile

For each page → load metadata

Generate page via diffusion inpainting

Assemble book

Return result

21. job_queue.py

(Optional, but recommended)

Handles background GPU jobs

Prevents overload

Scaling to multiple GPUs

22. config.py

Paths, global settings, model config.


1. mask_generator.py (critical for inpainting)
2. identity_profile.py (combines detection + appearance + embedding)
3. prompt_builder.py (turn appearance into prompts)
4. sdxl_pipeline_loader.py (big, essential module)
5. inpaint_generator.py (core generation pipeline)