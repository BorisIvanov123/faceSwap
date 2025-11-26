#!/bin/bash

# Setup script for Children's Book Generation Pipeline on Mac
# Run with: bash setup_mac.sh

echo "🚀 Setting up Children's Book Generation Pipeline for Mac"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch for Mac (MPS support)
echo ""
echo "Installing PyTorch with MPS support..."
pip install torch torchvision torchaudio

# Install InsightFace and dependencies
echo ""
echo "Installing InsightFace..."
pip install insightface
pip install onnxruntime  # For Mac CPU inference
pip install opencv-python

# Install Diffusers and transformers
echo ""
echo "Installing Diffusers ecosystem..."
pip install diffusers[torch]
pip install transformers
pip install accelerate
pip install safetensors

# Install IP-Adapter dependencies
echo ""
echo "Installing IP-Adapter..."
pip install git+https://github.com/tencent-ailab/IP-Adapter.git

# Install additional utilities
echo ""
echo "Installing utilities..."
pip install pillow
pip install numpy
pip install scipy
pip install tqdm

# Download InsightFace models
echo ""
echo "Downloading InsightFace models..."
python3 << EOF
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l")
print("✅ InsightFace models downloaded")
EOF

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p faces
mkdir -p templates
mkdir -p masks
mkdir -p output_faces
mkdir -p output_pages
mkdir -p models

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download IP-Adapter FaceID model:"
echo "   wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin -P models/"
echo ""
echo "2. (Optional) Download a book style LoRA from CivitAI"
echo ""
echo "3. Activate the environment: source venv/bin/activate"
echo "4. Run the pipeline!"