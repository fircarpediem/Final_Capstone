# 📓 Notebooks

Bộ Jupyter notebooks để training và testing trên Google Colab.

## 📁 Files

### 1. `colab_training.ipynb` - Training Notebook
**Mục đích:** Training và fine-tuning models trên Google Colab

**Chức năng:**
- ✅ Setup môi trường Colab (GPU, Drive mount)
- ✅ Install dependencies
- ✅ Load và prepare dataset
- ✅ Training loop với validation
- ✅ Evaluation trên public test
- ✅ Generate submission file
- ✅ Save checkpoints to Drive

**Cách dùng:**
1. Upload notebook lên Google Colab
2. Change Runtime → GPU (T4)
3. Upload dataset lên Drive tại `/MyDrive/RoadBuddy/`
4. Run all cells từ trên xuống
5. Monitor training progress
6. Download submission file

**Requirements:**
- Google Colab với GPU (T4 hoặc cao hơn)
- ~15GB Drive storage cho data + checkpoints
- Dataset uploaded to Drive

---

### 2. `colab_inference_demo.ipynb` - Inference & Demo Notebook
**Mục đích:** Testing và visualization không cần training

**Chức năng:**
- ✅ Load trained checkpoint
- ✅ Upload custom video để test
- ✅ Run single inference với visualization
- ✅ Batch inference trên test set
- ✅ Error analysis
- ✅ Compare single vs ensemble models
- ✅ Export results

**Cách dùng:**
1. Upload notebook lên Google Colab
2. Load checkpoint từ Drive
3. Upload video hoặc dùng sample từ dataset
4. Run inference cells
5. Visualize predictions
6. Export results

**Requirements:**
- Google Colab (CPU hoặc GPU)
- Trained checkpoint trong Drive
- Test videos

---

## 🚀 Quick Start

### Setup Google Drive Structure

Tạo cấu trúc folder trên Drive:

```
MyDrive/
└── RoadBuddy/
    ├── roadbuddy_winning_arch/    # Project code
    │   ├── src/
    │   ├── configs/
    │   └── requirements.txt
    ├── data/                       # Dataset
    │   └── raw/
    │       ├── train/
    │       │   ├── train.json
    │       │   └── videos/
    │       └── public_test/
    │           ├── public_test.json
    │           └── videos/
    ├── checkpoints/                # Model checkpoints
    ├── logs/                       # Training logs
    └── outputs/                    # Submission files
```

### Upload to Colab

**Cách 1: Upload notebook trực tiếp**
```python
from google.colab import files
uploaded = files.upload()
```

**Cách 2: Open from Drive**
1. Upload notebook vào Drive
2. Right-click → Open with → Google Colaboratory

**Cách 3: Clone từ GitHub**
```bash
!git clone https://github.com/YOUR_REPO/roadbuddy_winning_arch.git
```

---

## 💡 Tips & Tricks

### Training Tips

1. **GPU Selection:**
   - Runtime → Change runtime type → GPU → T4 (free)
   - V100/A100 nếu có Colab Pro

2. **Memory Management:**
   ```python
   # Trong config.yaml
   training:
     batch_size: 1
     gradient_accumulation_steps: 8  # Effective batch = 8
     fp16: true
     gradient_checkpointing: true
   ```

3. **Save Progress Frequently:**
   ```python
   # Auto-save to Drive every N steps
   save_steps: 500
   ```

4. **Use Tensorboard:**
   ```python
   %load_ext tensorboard
   %tensorboard --logdir /content/drive/MyDrive/RoadBuddy/logs
   ```

### Inference Tips

1. **Faster Inference:**
   ```python
   # Single model (nhanh hơn)
   cfg.model.vlm.mode = 'single'
   cfg.model.vlm.primary_model = 'qwen2vl'
   ```

2. **Best Accuracy:**
   ```python
   # Ensemble (chậm hơn nhưng chính xác hơn)
   cfg.model.vlm.mode = 'ensemble'
   ```

3. **Batch Processing:**
   ```python
   # Process nhiều videos cùng lúc
   for video in videos:
       result = pipeline.predict(video, question, choices)
   ```

### Common Issues

**Issue 1: Out of Memory**
```python
# Giảm batch size
config['training']['batch_size'] = 1
# Tăng gradient accumulation
config['training']['gradient_accumulation_steps'] = 16
# Enable gradient checkpointing
config['training']['gradient_checkpointing'] = True
```

**Issue 2: Colab Disconnect**
```python
# Keep session alive
import time
from IPython.display import display, Javascript

display(Javascript('''
  function KeepClicking(){
    console.log("Clicking");
    document.querySelector("colab-connect-button").click()
  }
  setInterval(KeepClicking, 60000)
'''))
```

**Issue 3: Drive Mount Issues**
```python
# Force remount
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)
```

---

## 📊 Expected Results

### Training (colab_training.ipynb)

**With T4 GPU:**
- Training time: ~6-8 hours (3 epochs)
- Memory usage: ~14GB GPU
- Expected val accuracy: 85-88% (single model)
- Expected val accuracy: 90-93% (ensemble)

**With V100 GPU:**
- Training time: ~3-4 hours (3 epochs)
- Memory usage: ~15GB GPU
- Same accuracy as T4

### Inference (colab_inference_demo.ipynb)

**Single Model Mode:**
- Inference time: ~5-8 seconds/video
- GPU memory: ~7GB
- Accuracy: 85-88%

**Ensemble Mode:**
- Inference time: ~10-15 seconds/video
- GPU memory: ~14GB
- Accuracy: 90-93%

---

## 🔗 Resources

**Google Colab:**
- Free tier: 12 hours session, T4 GPU
- Pro: Longer sessions, V100/A100 GPU
- Pro+: Priority access, more compute

**Drive Storage:**
- Free: 15GB
- Paid: 100GB+ plans available

**Helpful Links:**
- [Google Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colab Pro](https://colab.research.google.com/signup)
- [Colab Tips](https://colab.research.google.com/notebooks/snippets/advanced_outputs.ipynb)

---

## 📞 Support

Nếu gặp vấn đề:
1. Check cell outputs for error messages
2. Restart runtime: Runtime → Restart runtime
3. Clear outputs: Edit → Clear all outputs
4. Remount Drive if path issues
5. Check GPU availability: `!nvidia-smi`

---

**Happy Training! 🚗💨**
