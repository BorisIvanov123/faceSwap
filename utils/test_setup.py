#!/usr/bin/env python3
"""
Test script to validate the children's book generation setup.

Run this after installation to check that everything works.
"""

import sys
import subprocess
from pathlib import Path


def check_command(cmd, name):
    """Check if a command exists"""
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
        print(f"  ✅ {name}")
        return True
    except:
        print(f"  ❌ {name} not found")
        return False


def check_python_package(package, name=None):
    """Check if a Python package is installed"""
    if name is None:
        name = package
    try:
        __import__(package)
        print(f"  ✅ {name}")
        return True
    except ImportError:
        print(f"  ❌ {name} not installed")
        return False


def check_directories():
    """Check required directories exist"""
    dirs = ["faces", "templates", "masks", "output_faces", "output_pages", "models"]

    print("\n📁 Checking directories...")
    all_good = True
    for d in dirs:
        path = Path(d)
        if path.exists():
            print(f"  ✅ {d}/")
        else:
            print(f"  ⚠️  {d}/ does not exist (will be created)")
            path.mkdir(exist_ok=True)
            all_good = False

    return all_good


def test_insightface():
    """Test InsightFace installation"""
    print("\n🔍 Testing InsightFace...")

    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l")
        print("  ✅ InsightFace models downloaded")
        return True
    except Exception as e:
        print(f"  ❌ InsightFace error: {e}")
        return False


def test_torch():
    """Test PyTorch installation and device"""
    print("\n🔥 Testing PyTorch...")

    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")

        # Check MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            print("  ✅ MPS (Apple Silicon GPU) available")
            device = "mps"
        elif torch.cuda.is_available():
            print("  ✅ CUDA available")
            device = "cuda"
        else:
            print("  ⚠️  Using CPU (will be slower)")
            device = "cpu"

        print(f"  Using device: {device}")

        # Test tensor creation
        x = torch.randn(3, 3).to(device)
        print("  ✅ Can create tensors on device")

        return True

    except Exception as e:
        print(f"  ❌ PyTorch error: {e}")
        return False


def test_diffusers():
    """Test Diffusers installation"""
    print("\n🎨 Testing Diffusers...")

    try:
        from diffusers import StableDiffusionInpaintPipeline
        print("  ✅ Diffusers installed correctly")

        # Check if we can load a model (just check, don't download)
        print("  ℹ️  Note: First generation will download ~5GB of models")
        return True

    except Exception as e:
        print(f"  ❌ Diffusers error: {e}")
        return False


def test_face_detection():
    """Test face detection on a sample image"""
    print("\n👤 Testing face detection...")

    # Check if there's a test image
    test_images = list(Path("faces").glob("*.jpg")) + list(Path("faces").glob("*.jpeg"))

    if len(test_images) == 0:
        print("  ⚠️  No test images found in faces/ directory")
        print("     Add a photo to test face detection")
        return None

    test_image = test_images[0]
    print(f"  Testing with: {test_image.name}")

    try:
        import cv2
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=-1, det_size=(640, 640))

        img = cv2.imread(str(test_image))
        if img is None:
            print(f"  ❌ Could not load image")
            return False

        faces = app.get(img)

        if len(faces) > 0:
            print(f"  ✅ Detected {len(faces)} face(s)")
            face = faces[0]
            print(f"     Age: ~{int(face.age)}" if hasattr(face, 'age') else "")
            print(f"     Embedding shape: {face.normed_embedding.shape}")
            return True
        else:
            print("  ⚠️  No faces detected in image")
            return False

    except Exception as e:
        print(f"  ❌ Face detection error: {e}")
        return False


def test_mask_generation():
    """Test mask generation on templates"""
    print("\n🎭 Testing mask generation...")

    templates = list(Path("templates").glob("*.jpg")) + list(Path("templates").glob("*.png"))

    if len(templates) == 0:
        print("  ⚠️  No template images found in templates/ directory")
        return None

    template = templates[0]
    print(f"  Testing with: {template.name}")

    try:
        from improved_mask_generator import MaskGenerator

        generator = MaskGenerator()
        mask_path = Path("masks") / f"test_mask.png"

        generator.generate_smart_mask(
            str(template),
            str(mask_path),
            method="ellipse"
        )

        print(f"  ✅ Generated test mask: {mask_path}")
        return True

    except Exception as e:
        print(f"  ❌ Mask generation error: {e}")
        return False


def main():
    print("=" * 60)
    print("🧪 Children's Book Generator - System Check")
    print("=" * 60)

    all_good = True

    # System commands
    print("\n🖥️  Checking system commands...")
    all_good &= check_command("python3", "Python 3")
    all_good &= check_command("pip", "pip")

    # Python packages
    print("\n📦 Checking Python packages...")
    all_good &= check_python_package("torch", "PyTorch")
    all_good &= check_python_package("cv2", "OpenCV")
    all_good &= check_python_package("insightface", "InsightFace")
    all_good &= check_python_package("diffusers", "Diffusers")
    all_good &= check_python_package("transformers", "Transformers")
    all_good &= check_python_package("PIL", "Pillow")
    all_good &= check_python_package("numpy", "NumPy")

    # Directories
    check_directories()

    # Detailed tests
    all_good &= test_torch()
    all_good &= test_insightface()
    all_good &= test_diffusers()

    # Optional tests (if data available)
    test_face_detection()
    test_mask_generation()

    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou're ready to generate books! Try:")
        print("  python pipeline.py --child-photo faces/child.jpg --config example_book_config.json")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nPlease fix the issues above, then:")
        print("1. Run: bash setup_mac.sh")
        print("2. Run: python test_setup.py")
    print("=" * 60)


if __name__ == "__main__":
    main()