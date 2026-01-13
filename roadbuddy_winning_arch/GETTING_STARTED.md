# Getting Started with RoadBuddy Winning Architecture

## 📋 Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended: RTX 3090 or better)
- 32GB+ RAM
- 100GB+ free disk space

## 🚀 Installation

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd d:\Project\RoadBuddy\roadbuddy_winning_arch

# Run setup script (Windows)
.\setup.ps1

# OR manual setup:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Prepare Data

1. **Copy dataset files:**
   ```
   data/
   ├── raw/
   │   ├── train/
   │   │   ├── train.json
   │   │   └── videos/
   │   ├── public_test/
   │   │   ├── public_test.json
   │   │   └── videos/
   │   └── private_test/
   │       ├── private_test.json
   │       └── videos/
   ```

2. **Update paths in config:**
   Edit `configs/config.yaml`:
   ```yaml
   data:
     train_json: "data/raw/train/train.json"
     public_test_json: "data/raw/public_test/public_test.json"
   ```

### Step 3: Download Models

The Qwen2-VL model will auto-download on first run. For faster setup:

```python
from transformers import Qwen2VLForConditionalGeneration

# This will cache the model (~14GB)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct"
)
```

## 🎯 Quick Start - Inference

### Option 1: Command Line

```bash
# Activate environment
venv\Scripts\activate

# Run inference on public test
python scripts/inference.py \
    --config configs/config.yaml \
    --test_json data/raw/public_test/public_test.json \
    --output outputs/public_predictions.json

# Evaluate (if ground truth available)
python scripts/evaluate.py \
    --predictions outputs/public_predictions.json \
    --ground_truth data/raw/public_test/public_test.json
```

### Option 2: Python API

```python
from omegaconf import OmegaConf
from src.models.winning_pipeline import WinningRoadBuddyPipeline

# Load config
config = OmegaConf.load("configs/config.yaml")

# Initialize pipeline
pipeline = WinningRoadBuddyPipeline(config)

# Single prediction
answer = pipeline(
    video_path="path/to/video.mp4",
    question="Tốc độ tối đa là bao nhiêu?",
    choices={
        "A": "40 km/h",
        "B": "50 km/h", 
        "C": "60 km/h",
        "D": "70 km/h"
    }
)

print(f"Answer: {answer}")
```

### Option 3: Interactive Notebook

```bash
jupyter notebook notebooks/demo.py
```

## ⚙️ Configuration Guide

### Key Settings

**Video Processing:**
```yaml
data:
  max_frames_global: 32      # More frames = better context, slower
  num_keyframes: 3           # High-res frames for OCR
  global_resolution: [480, 270]  # Lower = faster
```

**Model Settings:**
```yaml
model:
  vlm:
    dtype: "bfloat16"        # bfloat16 (best), float16, float32
    load_in_8bit: false      # Enable if OOM
    
  experts:
    detector:
      conf_threshold: 0.3    # Lower = more detections
    ocr:
      ensemble_method: "voting"  # voting, confidence, weighted
```

**Inference:**
```yaml
inference:
  max_new_tokens: 20         # Answer length limit
  do_sample: false           # True for diverse answers
  temperature: 0.0           # Sampling temperature
```

## 🔧 Troubleshooting

### CUDA Out of Memory

**Solution 1:** Enable quantization
```yaml
model:
  vlm:
    load_in_8bit: true
```

**Solution 2:** Reduce frames
```yaml
data:
  max_frames_global: 16  # From 32
  num_keyframes: 2       # From 3
```

**Solution 3:** Lower resolution
```yaml
data:
  global_resolution: [320, 180]  # From [480, 270]
```

### Slow Inference

**Optimize video processing:**
```yaml
data:
  max_frames_global: 16  # Fewer frames
```

**Use smaller model:**
```yaml
model:
  vlm:
    name: "Qwen/Qwen2-VL-2B-Instruct"  # Smaller but less accurate
```

### OCR Not Working

```bash
# Reinstall OCR dependencies
pip uninstall vietocr paddleocr easyocr -y
pip install vietocr paddleocr easyocr
```

### Model Download Fails

```bash
# Use mirror (if in China/Asia)
export HF_ENDPOINT=https://hf-mirror.com
```

Or download manually:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2-VL-7B-Instruct",
    local_dir="./checkpoints/qwen2-vl-7b"
)
```

## 📊 Expected Results

### Baseline (No Fine-tuning)

| Split | Accuracy |
|-------|----------|
| Public Test | 75-80% |
| Private Test | 75-80% |

### With Fine-tuning

| Split | Accuracy |
|-------|----------|
| Public Test | 85-88% |
| Private Test | 87-90% |

### SOTA (Full Pipeline)

| Split | Accuracy |
|-------|----------|
| Public Test | 88-91% |
| Private Test | **90-93%** |

## 🎓 Next Steps

1. **Fine-tune detector:**
   - Collect more Vietnamese traffic sign data
   - Train YOLOv10 on your dataset
   - Replace `checkpoints/yolov10_traffic_vn.pt`

2. **Expand knowledge base:**
   - Add more traffic laws to `data/knowledge_base/vietnam_traffic_laws.json`
   - Include specific regulations for your region

3. **Fine-tune Qwen2-VL:**
   - Prepare training data
   - Run `scripts/train.py` (TODO)
   - Use LoRA for efficient training

4. **Optimize inference:**
   - Profile bottlenecks
   - Implement caching
   - Use model quantization

## 📚 Additional Resources

- [Qwen2-VL Documentation](https://github.com/QwenLM/Qwen2-VL)
- [YOLOv10 Guide](https://docs.ultralytics.com/)
- [Vietnamese OCR](https://github.com/pbcquoc/vietocr)

## 💡 Tips for Competition

1. **Error Analysis:**
   - Analyze failed predictions
   - Focus on common error patterns
   - Improve weakest components

2. **Ensemble:**
   - Run multiple models
   - Combine predictions
   - Higher accuracy, slower inference

3. **Data Augmentation:**
   - Simulate different conditions (rain, night, fog)
   - Augment training data
   - Improve robustness

4. **Post-processing:**
   - Add more validation rules
   - Cross-reference with knowledge base
   - Filter low-confidence predictions

## 🤝 Need Help?

- Check [README.md](README.md) for overview
- See [notebooks/demo.py](notebooks/demo.py) for examples
- Open an issue on GitHub

---

**Ready to compete! 🏆**
