# Complete Workflow Guide

This guide walks you through the entire pipeline from start to finish: preparing cloud images, generating synthetic rain, creating training data, training a model, and making predictions.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Workflow Overview](#workflow-overview)
3. [Step-by-Step Instructions](#step-by-step-instructions)
4. [Verification Steps](#verification-steps)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Windows, Linux, or macOS
- **Python**: 3.7 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended (NVIDIA with CUDA support)
- **Disk Space**: 2GB for dependencies + space for datasets

### Installation

1. **Clone or download the project**

2. **Create + activate a virtual environment (recommended):**

   **Windows (PowerShell):**
   ```bash
   py -m venv .venv
   .\\.venv\\Scripts\\Activate.ps1
   ```

   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. **For GPU support (recommended):**
   ```bash
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import PIL; from PIL import Image; print(f'Pillow: {PIL.__version__} | Has Resampling: {hasattr(Image, \'Resampling\')}')"
   ```

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAIN DETECTION PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

INPUT: Cloud Images (any size)
   │
   ├─── STEP 1: Collect Cloud Images
   │    └─► Place in project root (e.g., 6_cumulonimbus_000005.jpg)
   │
   ├─── STEP 2: Generate Synthetic Rain Masks
   │    └─► Run: python rain_mask_generator.py
   │    └─► Output: output/ folder with 5 rain masks (224×224 PNG)
   │
   ├─── STEP 3: Overlay Rain on Cloud Images
   │    └─► Run: python rain_overlay.py
   │    └─► Output: overlayed_images/ folder (rainy versions, 224×224 JPG)
   │
   ├─── STEP 4: Prepare Training Dataset
   │    └─► Rain class: overlayed_images/ (label = 1)
   │    └─► No-rain class: original images (label = 0)
   │    └─► Automatic 80/20 train/validation split
   │
   ├─── STEP 5: Train Classification Model
   │    └─► Run: python train_rain_classifier.py
   │    └─► Output: best_rain_classifier.pth, training_history.png
   │
   └─── STEP 6: Make Predictions
        └─► Run: python predict_rain.py --image path/to/image.jpg
        └─► Output: Rain/No Rain + confidence score

OUTPUT: Trained model that detects rain in images
```

## Step-by-Step Instructions

### Step 1: Collect Cloud Images

**Objective**: Gather base images of clouds (without rain) for training.

**Instructions:**

1. **Collect images** of clouds, sky, or weather conditions without rain
2. **Place images** in the project root directory:
   ```
   rain-overlay-filter/
   ├── 6_cumulonimbus_000005.jpg
   ├── cloud_photo_1.jpg
   ├── cloud_photo_2.jpg
   └── ...
   ```

**Recommendations:**
- **Quantity**: At least 10-20 different cloud images
- **Variety**: Different cloud types, lighting, times of day
- **Quality**: Good resolution (will be resized to 224×224)
- **Format**: JPG, PNG, or BMP

**Alternative**: Create a `no-rain/` folder for organization:
```
rain-overlay-filter/
├── no-rain/
│   ├── cloud1.jpg
│   ├── cloud2.jpg
│   └── ...
```

**Verification:**
```bash
# Count images in current directory
ls *.jpg *.png | wc -l
```

---

### Step 2: Generate Synthetic Rain Masks

**Objective**: Create realistic rain patterns to overlay on cloud images.

**Command:**
```bash
python rain_mask_generator.py
```

**What happens:**
1. Script generates 5 different rain mask types
2. All masks are 224×224 pixels, grayscale PNG
3. Saves to `output/` folder

**Output Files:**
```
output/
├── light_rain_topdown_224x224.png       # Light rain effect
├── heavy_rain_topdown_224x224.png       # Heavy downpour
├── rain_drops_224x224.png               # Circular drops
├── combined_rain_topdown_224x224.png    # Streaks + drops
└── noise_rain_224x224.png               # Noise-based rain
```

**Customization** (optional):

Edit `rain_mask_generator.py` main() function to create custom masks:

```python
# Add your own custom rain pattern
custom_rain = generator.generate_rain_streaks(
    num_streaks=80,
    min_length=25,
    max_length=45,
    thickness=2,
    intensity=200,
    top_down=True
)
generator.save_mask(custom_rain, 'output/custom_rain.png')
```

**Verification:**
```bash
ls output/*.png
# Should show 5 (or more) PNG files
```

**Visual Check**: Open masks in image viewer - should see white rain patterns on black background

---

### Step 3: Overlay Rain on Cloud Images

**Objective**: Combine cloud images with rain masks to create rainy versions.

**Command:**
```bash
python rain_overlay.py
```

**What happens:**
1. Finds all images in current directory (cloud images)
2. Finds all masks in `output/` folder
3. Creates all combinations (N images × M masks)
4. Resizes everything to 224×224
5. Applies rain overlay with 70% opacity
6. Saves results to `overlayed_images/` folder

**Output:**

For each combination of cloud image and rain mask:
```
overlayed_images/
├── 6_cumulonimbus_000005_light_rain_topdown_224x224.jpg
├── 6_cumulonimbus_000005_heavy_rain_topdown_224x224.jpg
├── 6_cumulonimbus_000005_rain_drops_224x224.jpg
├── 6_cumulonimbus_000005_combined_rain_topdown_224x224.jpg
├── 6_cumulonimbus_000005_noise_rain_224x224.jpg
└── ... (all combinations)
```

**Example Console Output:**
```
Creating rain overlays...
==================================================
Found 2 images and 5 masks
Will generate 10 overlays

Created: 6_cumulonimbus_000005_light_rain_topdown_224x224.jpg
Created: 6_cumulonimbus_000005_heavy_rain_topdown_224x224.jpg
...
==================================================
Done! Generated 10 images in 'overlayed_images/' folder
All images are 224x224 pixels
```

**Customization** (optional):

Edit `rain_overlay.py` to change opacity or blend mode:

```python
# In main() function, change these parameters:
results = overlay.batch_process_folder(
    image_folder='.',
    mask_folder='output',
    output_dir='overlayed_images',
    blend_mode='add',      # Try 'screen' or 'lighten'
    opacity=0.7            # Try 0.5-0.9
)
```

**Verification:**
```bash
ls overlayed_images/*.jpg | wc -l
# Should show (number of cloud images × number of rain masks)
```

**Visual Check**: Open overlayed images - should see clouds with visible rain

---

### Step 4: Prepare Training Dataset

**Objective**: Organize data into rain and no-rain classes.

**Automatic Setup** (handled by training script):

The training script automatically:
1. Uses images in `overlayed_images/` as **rain class** (label = 1)
2. Uses original images in root as **no-rain class** (label = 0)
3. Splits data 80% training, 20% validation (stratified)

**Manual Check**:

Ensure you have:
- ✅ Rain images in `overlayed_images/`
- ✅ Original cloud images in root directory (or `no-rain/` folder)
- ✅ Roughly balanced counts (similar number of each class)

**Expected Structure:**
```
rain-overlay-filter/
├── 6_cumulonimbus_000005.jpg        ← No-rain class
├── cloud_photo_1.jpg                 ← No-rain class
├── overlayed_images/
│   ├── ..._heavy_rain_topdown.jpg   ← Rain class
│   ├── ..._light_rain_topdown.jpg   ← Rain class
│   └── ...                           ← Rain class
```

**Verification:**
```bash
# Count no-rain images
ls *.jpg | wc -l

# Count rain images
ls overlayed_images/*.jpg | wc -l

# Ideally, counts should be similar
```

---

### Step 5: Train Classification Model

**Objective**: Train MobileNetV4 to distinguish rain from no-rain.

**Command:**
```bash
python train_rain_classifier.py
```

**What happens:**

1. **Dataset preparation**:
   - Loads rain images from `overlayed_images/`
   - Loads no-rain images from root directory
   - Splits 80/20 train/validation (stratified)

2. **Model creation**:
   - Loads MobileNetV4 (conv_medium variant) pretrained on ImageNet via timm library
   - Automatically sets final layer for binary classification (2 classes)

3. **Training** (10 epochs):
   - Applies data augmentation (flips, rotations)
   - Trains with Adam optimizer (lr=0.001)
   - Evaluates on validation set each epoch
   - Saves best model (highest validation accuracy)

4. **Output**:
   - `best_rain_classifier.pth` - Model weights
   - `training_history.png` - Loss/accuracy plots

**Console Output:**
```
Rain Binary Classification with MobileNetV4
============================================================

Using device: cuda

Preparing dataset...
Found 10 rain images
Found 2 no-rain images
Total dataset size: 12 images

Train set: 9 images
Validation set: 3 images

Creating MobileNetV4 model...

Starting training...
============================================================

Epoch [1/10]
--------------------------------------------------
Batch [1/1], Loss: 0.6892, Acc: 55.56%
Train Loss: 0.6892, Train Acc: 55.56%
Val Loss: 0.6534, Val Acc: 66.67%
Saved best model with validation accuracy: 66.67%

Epoch [2/10]
--------------------------------------------------
...

Epoch [10/10]
--------------------------------------------------
Train Loss: 0.0234, Train Acc: 100.00%
Val Loss: 0.0456, Val Acc: 100.00%
Saved best model with validation accuracy: 100.00%

============================================================
Training completed!
Best model saved as 'best_rain_classifier.pth'
Training history plot saved as 'training_history.png'
```

**Training Time:**
- GPU (CUDA): 5-10 minutes
- CPU: 30-60 minutes

**Verification:**
```bash
# Check model file exists
ls -lh best_rain_classifier.pth
# Should be ~15-20 MB

# Check training plot
ls -lh training_history.png
# Open in image viewer to see loss/accuracy curves
```

**Interpreting Results:**

Open `training_history.png`:
- **Left plot (Loss)**: Should decrease over epochs
- **Right plot (Accuracy)**: Should increase over epochs

**Good training:**
- Validation accuracy reaches 90-100%
- Train and validation curves are close (no major gap)

**Overfitting** (needs more data):
- Train accuracy high (>95%)
- Validation accuracy low (<70%)
- Large gap between train and val curves

---

### Step 6: Make Predictions

**Objective**: Use trained model to detect rain in new images.

**Command:**
```bash
python predict_rain.py --image path/to/image.jpg
```

**Examples:**

Test on a rainy image:
```bash
python predict_rain.py --image overlayed_images/6_cumulonimbus_000005_heavy_rain_topdown_224x224.jpg
```

Expected output:
```
Using device: cuda
Loading model from best_rain_classifier.pth...
Analyzing image: overlayed_images/6_cumulonimbus_000005_heavy_rain_topdown_224x224.jpg

==================================================
Prediction: Rain
Confidence: 98.45%
==================================================
```

Test on a no-rain image:
```bash
python predict_rain.py --image 6_cumulonimbus_000005.jpg
```

Expected output:
```
==================================================
Prediction: No Rain
Confidence: 99.12%
==================================================
```

**Options:**
```bash
# Use specific model
python predict_rain.py --image test.jpg --model my_model.pth

# Force CPU (no GPU)
python predict_rain.py --image test.jpg --device cpu
```

**Batch Prediction** (programmatic):

```python
from predict_rain import load_model, predict_image
import torch
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('best_rain_classifier.pth', device)

# Predict on multiple images
for img_path in Path('test_images').glob('*.jpg'):
    pred, conf, name = predict_image(model, str(img_path), device)
    print(f"{img_path.name}: {name} ({conf:.1f}%)")
```

---

## Verification Steps

### After Each Step

#### After Step 2 (Rain Masks):
```bash
# Check output folder
ls output/

# Expected: 5 PNG files
# Visual: Open in image viewer - white rain on black background
```

#### After Step 3 (Overlay):
```bash
# Check overlayed images
ls overlayed_images/

# Expected: N × M images (N clouds × M masks)
# Visual: Should see clouds with rain overlay
```

#### After Step 5 (Training):
```bash
# Check model file
ls -lh best_rain_classifier.pth

# Check training plot
open training_history.png  # macOS
xdg-open training_history.png  # Linux
start training_history.png  # Windows
```

#### After Step 6 (Prediction):
Test on both rain and no-rain images - should get correct predictions with high confidence (>90%)

---

## Troubleshooting

### Common Issues

#### Issue: `AttributeError: module 'PIL.Image' has no attribute 'Resampling'`

**Problem**: You’re running with an older Pillow (common when using system Python instead of the project venv). This project uses `Image.Resampling.LANCZOS`.

**Solutions**:
1. Activate the project virtual environment, then reinstall requirements:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Or upgrade Pillow in the interpreter you’re using:
   ```bash
   python -m pip install -U pillow
   ```
3. Verify:
   ```bash
   python -c "import PIL; from PIL import Image; print(PIL.__version__, hasattr(Image,'Resampling'))"
   ```

#### Issue: "No images found"

**Problem**: `rain_overlay.py` can't find cloud images

**Solution**:
```bash
# Check current directory
ls *.jpg *.png

# If images are elsewhere, edit rain_overlay.py:
# Change image_folder='.' to image_folder='no-rain'
```

#### Issue: "CUDA out of memory"

**Problem**: GPU doesn't have enough memory for training

**Solutions**:
1. Reduce batch size in `train_rain_classifier.py`:
   ```python
   batch_size = 8  # or even 4
   ```

2. Use CPU instead:
   ```python
   device = 'cpu'
   ```

3. Use smaller model:
   ```python
   import timm

   model = timm.create_model(
       'mobilenetv4_conv_small.e500_r224_in1k',
       pretrained=True,
       num_classes=2
   )
   ```

#### Issue: Low validation accuracy (<70%)

**Problem**: Model not learning well

**Solutions**:
1. **More data**: Add more diverse cloud images
2. **More epochs**: Change `num_epochs = 20`
3. **Check labels**: Verify rain/no-rain images are correct
4. **Balance data**: Equal counts of rain and no-rain images

#### Issue: Overfitting (high train, low val accuracy)

**Problem**: Model memorizing training data

**Solutions**:
1. **More training data**: Collect more images
2. **More augmentation**: Add color jitter, brightness variations
3. **Regularization**: Add dropout to model

#### Issue: "Model file not found"

**Problem**: Trying to predict before training

**Solution**:
```bash
# Train model first
python train_rain_classifier.py

# Then predict
python predict_rain.py --image test.jpg
```

#### Issue: Predictions always same class

**Problem**: Imbalanced dataset or model not trained properly

**Solutions**:
1. Check dataset balance:
   ```bash
   ls overlayed_images/*.jpg | wc -l  # Rain count
   ls *.jpg | wc -l  # No-rain count
   ```

2. Retrain with balanced data
3. Check training curves in `training_history.png`

---

## Next Steps

### Improving the Model

1. **Collect more data**:
   - More diverse cloud types
   - Different weather conditions
   - Various times of day

2. **Create more rain variations**:
   - Edit `rain_mask_generator.py`
   - Add custom rain patterns
   - Vary intensity and density

3. **Fine-tune training**:
   - Experiment with learning rates
   - Try longer training (20-30 epochs)
   - Add learning rate scheduling

4. **Advanced techniques**:
   - Ensemble multiple models
   - Use test-time augmentation
   - Export to ONNX for production

### Production Deployment

1. **Model optimization**:
   - Convert to TorchScript
   - Quantization for faster inference
   - ONNX export for cross-platform

2. **API deployment**:
   - Flask/FastAPI web service
   - Docker containerization
   - Cloud deployment (AWS, GCP, Azure)

3. **Mobile deployment**:
   - PyTorch Mobile
   - TensorFlow Lite conversion
   - Core ML for iOS

---

## Summary

**Complete Workflow:**

1. ✅ Prepare cloud images → Root directory
2. ✅ Generate rain masks → `python rain_mask_generator.py`
3. ✅ Overlay rain → `python rain_overlay.py`
4. ✅ Train model → `python train_rain_classifier.py`
5. ✅ Make predictions → `python predict_rain.py --image path.jpg`

**Key Files:**
- Input: Cloud images (any size)
- Output: `best_rain_classifier.pth` (trained model)
- Prediction: Rain/No Rain + confidence

**Time Required:**
- Setup: 10 minutes
- Data preparation: 5-10 minutes
- Training: 5-60 minutes (GPU/CPU)
- Total: ~30-90 minutes for complete pipeline

**Expected Results:**
- Validation accuracy: >95% with good data
- Inference: <50ms per image (GPU)
- Model size: ~15-20 MB
