# Rain Detection System

**Complete end-to-end pipeline for synthetic rain generation and binary rain classification using deep learning.**

This project provides a comprehensive workflow for:
1. 🌧️ **Generating synthetic rain masks** with realistic patterns
2. 🖼️ **Overlaying rain onto cloud images** to create training data  
3. 🤖 **Training a MobileNetV4 classifier** to detect rain in images
4. 🔍 **Making predictions** on new images

## Project Overview

This system creates synthetic rain data and trains a binary classifier to distinguish between rainy and non-rainy conditions in cloud/weather imagery. The entire pipeline is designed for 224×224 pixel images, optimized for efficiency and accuracy.

## Complete Workflow

```
Step 1: Cloud Images               →  Step 2: Generate Rain Masks
(cumulonimbus clouds)                  (synthetic rain patterns)
                                       
        ↓                                      ↓
                                       
Step 3: Overlay Rain               →  Step 4: Train Classifier
(combine clouds + rain)                (MobileNetV4 transfer learning)
                                       
        ↓                                      ↓
                                       
Training Dataset                   →  Step 5: Predict
(rain + no-rain images)                (detect rain in new images)
```

## Key Features

### Rain Generation
- **Top-down perspective**: Radial rain streaks (camera facing up)
- **Multiple rain types**: Streaks, drops, combined, noise-based
- **Customizable**: Intensity, density, angle, thickness

### Rain Overlay
- **Automatic resizing**: All images scaled to 224×224
- **Multiple blend modes**: Add, screen, lighten
- **Batch processing**: Apply all masks to all images

### Classification Model
- **Transfer learning**: MobileNetV4 pretrained on ImageNet (via timm library)
- **Binary classification**: Rain vs. No Rain
- **Efficient**: Fast training and inference
- **Accurate**: >95% validation accuracy with sufficient data

## Installation

### Requirements

- Python 3.9+ (recommended: 3.10+)
- PyTorch (with CUDA support recommended)
- See `requirements.txt` for all dependencies

### Install Dependencies

**Recommended (Windows/macOS/Linux): use a virtual environment** so you consistently use the same Pillow/PyTorch versions.

**Windows (PowerShell):**
```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS/Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```bash
python -m pip install -r requirements.txt
```

This installs:
- **torch**: PyTorch framework
- **torchvision**: Computer vision utilities
- **timm**: PyTorch Image Models (for MobileNetV4)
- **pillow**: Image processing
- **matplotlib**: Visualization
- **scikit-learn**: Data splitting utilities

**For GPU support** (recommended for training):
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Troubleshooting

### `AttributeError: module 'PIL.Image' has no attribute 'Resampling'`

**Why it happens**: this project uses `Image.Resampling.LANCZOS`. If you run outside the virtual environment, your system Python may be importing an older Pillow that doesn’t include `Image.Resampling`.

**Fix**:
- Activate the project venv and re-run.
- Or upgrade Pillow in the interpreter you’re using: `python -m pip install -U pillow`

**Quick check**:
```bash
python -c "import PIL; from PIL import Image; print('Pillow:', PIL.__version__); print('Has Resampling:', hasattr(Image, 'Resampling'))"
```

## Quick Start

Follow these steps to generate data, train a model, and make predictions:

### Step 1: Prepare Cloud Images

Place your cloud images (without rain) in the project root directory:
```
rain-overlay-filter/
├── 6_cumulonimbus_000005.jpg    # Example cloud image
├── cloud_image_2.jpg
└── ...
```

**Tip**: You can also create a `no-rain/` folder for organized storage.

### Step 2: Generate Synthetic Rain Masks

Generate rain mask patterns:

```bash
python rain_mask_generator.py
```

**Output**: Creates 5 rain masks in `output/` folder
- `light_rain_topdown_224x224.png`
- `heavy_rain_topdown_224x224.png`
- `rain_drops_224x224.png`
- `combined_rain_topdown_224x224.png`
- `noise_rain_224x224.png`

### Step 3: Overlay Rain on Cloud Images

Apply rain masks to cloud images:

```bash
python rain_overlay.py
```

**Output**: Creates rainy images in `overlayed_images/` folder
- Combines each cloud image with each rain mask
- All images resized to 224×224
- Example: `6_cumulonimbus_000005_heavy_rain_topdown_224x224.jpg`

### Step 4: Train the Classifier

Train MobileNetV4 to detect rain:

```bash
python train_rain_classifier.py
```

**Process**:
- Uses images from `overlayed_images/` as rain class (label=1)
- Uses original cloud images as no-rain class (label=0)
- Automatically splits into 80% train / 20% validation
- Trains for 10 epochs with data augmentation
- Saves best model as `best_rain_classifier.pth`

**Output**:
- `best_rain_classifier.pth` - Trained model weights
- `training_history.png` - Loss and accuracy plots

### Step 5: Make Predictions

Classify new images:

```bash
python predict_rain.py --image path/to/image.jpg
```

**Example**:
```bash
python predict_rain.py --image overlayed_images/6_cumulonimbus_000005_heavy_rain_topdown_224x224.jpg
```

**Output**:
```
==================================================
Prediction: Rain
Confidence: 98.45%
==================================================
```

## Detailed Documentation

For in-depth information about each component, see the `docs/` folder:

- **[rain_mask_generator.md](docs/rain_mask_generator.md)** - Rain mask generation (Step 2)
- **[rain_overlay.md](docs/rain_overlay.md)** - Overlaying rain on images (Step 3)
- **[train_rain_classifier.md](docs/train_rain_classifier.md)** - Model training (Step 4)
- **[predict_rain.md](docs/predict_rain.md)** - Making predictions (Step 5)

## Project Structure

```
rain-overlay-filter/
├── rain_mask_generator.py      # Generate synthetic rain masks
├── rain_overlay.py              # Apply rain to cloud images
├── train_rain_classifier.py    # Train classification model
├── predict_rain.py              # Make predictions on images
├── requirements.txt             # Python dependencies
│
├── docs/                        # Detailed documentation
│   ├── rain_mask_generator.md
│   ├── rain_overlay.md
│   ├── train_rain_classifier.md
│   └── predict_rain.md
│
├── output/                      # Generated rain masks
│   ├── light_rain_topdown_224x224.png
│   ├── heavy_rain_topdown_224x224.png
│   └── ...
│
├── overlayed_images/            # Rain + cloud combinations
│   ├── cloud1_heavy_rain.jpg
│   └── ...
│
├── best_rain_classifier.pth    # Trained model weights
└── training_history.png         # Training metrics plot
```

## Usage Examples

### Generate Custom Rain Mask

```python
from rain_mask_generator import RainMaskGenerator

generator = RainMaskGenerator(size=(224, 224))

# Heavy storm
storm = generator.generate_rain_streaks(
    num_streaks=150,
    max_length=60,
    thickness=2,
    intensity=230,
    top_down=True
)
generator.save_mask(storm, 'output/storm.png')
```

### Overlay Rain on Image

```python
from rain_overlay import RainOverlay
from PIL import Image

overlay = RainOverlay(target_size=(224, 224))

base = Image.open('cloud.jpg')
rain = Image.open('output/heavy_rain.png')

result = overlay.apply_overlay(base, rain, blend_mode='add', opacity=0.7)
result.save('rainy_cloud.jpg')
```

## Technical Details

### Rain Mask Generation
- **Top-down perspective**: Radial streaks from center (simulates camera facing up)
- **Randomization**: Position, length, intensity vary per element
- **Grayscale output**: 0-255 pixel values

### Image Overlay
- **Blend modes**: Add, screen, lighten
- **Automatic resizing**: All images scaled to 224×224
- **High quality**: JPEG quality=95

### Classification Model
- **Architecture**: MobileNetV4 (conv_medium)
- **Pretrained**: ImageNet weights (via timm)
- **Input**: 224×224×3 RGB images
- **Output**: 2 classes (No Rain=0, Rain=1)
- **Training**: Adam optimizer, CrossEntropyLoss
- **Augmentation**: Random flips, rotations

### Performance
- **Training time**: 5-10 minutes (GPU), 30-60 min (CPU)
- **Inference**: <50ms per image (GPU), ~200ms (CPU)
- **Accuracy**: >95% with sufficient diverse data

## Model Architecture

Based on: [MobileNetV4: Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)

**MobileNetV4 Highlights:**
- Improved efficiency vs. MobileNetV3
- Better accuracy-to-latency tradeoff
- Multiple variants (conv / hybrid)
- Optimized for mobile/edge devices

**Transfer Learning:**
```
ImageNet (1000 classes) → Modified classifier → Rain detection (2 classes)
```

## Dependencies

```
numpy - Array operations
pillow - Image manipulation
torch - Deep learning framework
timm - Pretrained models (MobileNetV4)
torchvision - Transforms and utilities
matplotlib - Plotting training curves
scikit-learn - Train/test splitting
```

## Tips and Best Practices

### For Better Results

1. **Diverse Data**: Use various cloud types and lighting conditions
2. **Balance Dataset**: Equal rain and no-rain samples
3. **Quality Images**: High-resolution source images before resize
4. **Multiple Masks**: Create variety in rain patterns
5. **Validate**: Always check training plots for overfitting

### Common Issues

**Low accuracy?**
- Increase training data
- Train for more epochs
- Check data labels
- Review augmentation settings

**CUDA out of memory?**
- Reduce batch size to 8
- Use MobileNetV4 conv_small instead
- Train on CPU (slower)

**Rain too subtle/strong?**
- Adjust opacity in overlay (0.5-0.9)
- Change rain mask intensity
- Try different blend modes

## Use Cases

- 🌦️ Weather classification systems
- 📊 Climate data analysis
- 🚗 Autonomous vehicle perception (rain detection)
- 📱 Mobile weather apps
- 🔬 Research in rain removal algorithms
- 🎓 Educational projects in deep learning
- 🖼️ Synthetic data generation for ML

## Citation

If you use this project in your research, please reference:

```bibtex
@software{rain_detection_system,
  title={Rain Detection System: Synthetic Data Generation and Classification},
  author={Rain Overlay Filter},
  year={2026},
  url={https://github.com/your-repo/rain-overlay-filter}
}
```

## Contributing

Contributions welcome! Areas for improvement:
- Additional rain patterns (drizzle, sleet, snow)
- More blend modes
- Multi-class classification (light/moderate/heavy rain)
- Real-time video processing
- Mobile deployment optimization

## License

MIT License

## Acknowledgments

- MobileNetV4 implementation via timm (PyTorch Image Models)
- Inspired by weather augmentation techniques in computer vision
- Transfer learning approach from ImageNet pretrained models

### Programmatic Prediction

```python
from predict_rain import load_model, predict_image
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('best_rain_classifier.pth', device)

prediction, confidence, class_name = predict_image(
    model,
    'test_image.jpg',
    device
)

print(f"{class_name}: {confidence:.2f}%")
# Output: "Rain: 95.32%"
```

## Use Cases

- Data augmentation for computer vision models
- Image preprocessing for rain removal algorithms
- Synthetic dataset generation
- Weather effect simulation
- Training denoising networks

## License

MIT License
