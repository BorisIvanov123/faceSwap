1. Go to your project folder:
cd /workspace/diffusion

2. Activate the venv:
source venv/bin/activate


When it succeeds, your shell prompt will look like:

(venv) root@...


This means you are now inside your AI environment and can run anything (diffusers, torch, IP-Adapter, ControlNet, etc.)


pip list | grep -E "diffusers|transformers|control|adapter|lora|clip|timm|einops|gdown|safetensors|accelerate"
