"""
Test Installation Script
Verify that all components are properly installed
"""

import sys
from pathlib import Path

print("="*70)
print("RoadBuddy Winning Architecture - Installation Test")
print("="*70)
print()

# Test 1: Python version
print("[1/10] Testing Python version...")
if sys.version_info >= (3, 10):
    print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
else:
    print(f"  ✗ Python 3.10+ required (found {sys.version_info.major}.{sys.version_info.minor})")
    sys.exit(1)

# Test 2: PyTorch
print("\n[2/10] Testing PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠ CUDA not available (CPU only)")
except ImportError:
    print("  ✗ PyTorch not installed")
    sys.exit(1)

# Test 3: Transformers
print("\n[3/10] Testing Transformers...")
try:
    import transformers
    print(f"  ✓ Transformers {transformers.__version__}")
except ImportError:
    print("  ✗ Transformers not installed")
    sys.exit(1)

# Test 4: Ultralytics (YOLO)
print("\n[4/10] Testing Ultralytics...")
try:
    from ultralytics import YOLO
    print(f"  ✓ Ultralytics installed")
except ImportError:
    print("  ✗ Ultralytics not installed")
    sys.exit(1)

# Test 5: OCR libraries
print("\n[5/10] Testing OCR libraries...")
ocr_count = 0

try:
    from vietocr.tool.predictor import Predictor
    print("  ✓ VietOCR")
    ocr_count += 1
except ImportError:
    print("  ⚠ VietOCR not available")

try:
    from paddleocr import PaddleOCR
    print("  ✓ PaddleOCR")
    ocr_count += 1
except ImportError:
    print("  ⚠ PaddleOCR not available")

try:
    import easyocr
    print("  ✓ EasyOCR")
    ocr_count += 1
except ImportError:
    print("  ⚠ EasyOCR not available")

if ocr_count == 0:
    print("  ✗ No OCR libraries installed")
    sys.exit(1)

# Test 6: Sentence Transformers
print("\n[6/10] Testing Sentence Transformers...")
try:
    from sentence_transformers import SentenceTransformer
    print("  ✓ Sentence Transformers installed")
except ImportError:
    print("  ✗ Sentence Transformers not installed")
    sys.exit(1)

# Test 7: OpenCV
print("\n[7/10] Testing OpenCV...")
try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except ImportError:
    print("  ✗ OpenCV not installed")
    sys.exit(1)

# Test 8: OmegaConf
print("\n[8/10] Testing OmegaConf...")
try:
    from omegaconf import OmegaConf
    print("  ✓ OmegaConf installed")
except ImportError:
    print("  ✗ OmegaConf not installed")
    sys.exit(1)

# Test 9: Project modules
print("\n[9/10] Testing project modules...")
try:
    sys.path.append(str(Path(__file__).parent))
    from src.data.video_processor import VideoProcessor
    from src.experts.detector import TrafficDetector
    from src.experts.ocr_ensemble import OCREnsemble
    from src.experts.knowledge_base import TrafficKnowledgeBase
    from src.models.winning_pipeline import WinningRoadBuddyPipeline
    print("  ✓ All project modules imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import project modules: {e}")
    sys.exit(1)

# Test 10: Configuration
print("\n[10/10] Testing configuration...")
try:
    config_path = Path(__file__).parent / "configs" / "config.yaml"
    if config_path.exists():
        config = OmegaConf.load(config_path)
        print("  ✓ Configuration loaded")
    else:
        print("  ✗ Configuration file not found")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Failed to load configuration: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("✓ All tests passed!")
print("="*70)
print()
print("Installation successful! You can now:")
print("  1. Place your data in data/raw/")
print("  2. Run inference with: python scripts/inference.py")
print("  3. Check the demo: python notebooks/demo.py")
print()
print("For more information, see GETTING_STARTED.md")
print()
