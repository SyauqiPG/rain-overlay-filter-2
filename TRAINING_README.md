# Rain Binary Classification with MobileNetV3

Binary classification model to detect rain in images using MobileNetV3 transfer learning.

## Model Architecture

- **Base Model**: MobileNetV3-Large (pretrained on ImageNet)
- **Transfer Learning**: Modified final classification layer for binary output
- **Input Size**: 224x224 pixels
- **Classes**: 
  - 0: No Rain
  - 1: Rain

## Reference

Implementation based on: [Understanding and Implementing MobileNetV3](https://medium.com/@RobuRishabh/understanding-and-implementing-mobilenetv3-422bd0bdfb5a)

## Installation

```bash
pip install -r requirements.txt
```

**Note**: For PyTorch with CUDA support, install from [pytorch.org](https://pytorch.org/):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Dataset Preparation

The training script expects:
- **Rain images**: Images with rain overlay (from `overlayed_images/` folder)
- **No-rain images**: Original images without rain

The dataset is automatically split into:
- Training set: 80%
- Validation set: 20%

## Training

Run the training script:

```bash
python train_rain_classifier.py
```

### Training Parameters

- **Epochs**: 10
- **Batch Size**: 16
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Data Augmentation**: 
  - Random horizontal flip
  - Random rotation (±10°)
  - ImageNet normalization

### Output Files

- `best_rain_classifier.pth`: Best model weights (highest validation accuracy)
- `training_history.png`: Training/validation loss and accuracy plots

## Inference

Predict rain in a single image:

```bash
python predict_rain.py --image path/to/image.jpg
```

**Options:**
- `--image`: Path to image (required)
- `--model`: Path to model weights (default: `best_rain_classifier.pth`)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)

### Example

```bash
python predict_rain.py --image overlayed_images/6_cumulonimbus_000005_heavy_rain_topdown_224x224.jpg
```

Output:
```
Using device: cuda
Loading model from best_rain_classifier.pth...
Analyzing image: overlayed_images/6_cumulonimbus_000005_heavy_rain_topdown_224x224.jpg

==================================================
Prediction: Rain
Confidence: 98.45%
==================================================
```

## Project Structure

```
rain-overlay-filter/
├── rain_mask_generator.py       # Generate rain masks
├── rain_overlay.py               # Apply rain overlays to images
├── train_rain_classifier.py     # Train classification model
├── predict_rain.py               # Inference script
├── requirements.txt              # Python dependencies
├── output/                       # Generated rain masks
├── overlayed_images/             # Images with rain overlay
├── best_rain_classifier.pth     # Trained model weights
└── training_history.png          # Training plots
```

## Model Performance

The model uses transfer learning from MobileNetV3-Large, which provides:
- Fast inference (optimized for mobile/edge devices)
- High accuracy with limited training data
- Efficient architecture with depthwise separable convolutions

Expected performance with sufficient data:
- Validation Accuracy: >95%
- Inference Time: <50ms per image (GPU)

## Custom Training

Modify training parameters in `train_rain_classifier.py`:

```python
# Change number of epochs
num_epochs = 20

# Change batch size
batch_size = 32

# Change learning rate
lr = 0.0001

# Use MobileNetV3-Small instead
model = models.mobilenet_v3_small(pretrained=True)
```

## Programmatic Usage

```python
from train_rain_classifier import create_model
from predict_rain import load_model, predict_image
import torch

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('best_rain_classifier.pth', device)

# Predict
prediction, confidence, class_name = predict_image(
    model, 
    'path/to/image.jpg', 
    device
)

print(f"{class_name}: {confidence:.2f}%")
```

## Tips for Better Accuracy

1. **More Training Data**: Collect more diverse rain/no-rain images
2. **Data Augmentation**: Add color jitter, brightness adjustments
3. **Fine-tuning**: Unfreeze earlier layers for domain-specific features
4. **Ensemble**: Combine predictions from multiple models
5. **Longer Training**: Increase epochs with learning rate scheduling

## Troubleshooting

**CUDA Out of Memory**:
- Reduce batch size
- Use `mobilenet_v3_small` instead of `large`
- Use CPU: `--device cpu`

**Low Accuracy**:
- Check dataset balance (equal rain/no-rain samples)
- Increase training data
- Train for more epochs
- Adjust learning rate

## License

MIT License
