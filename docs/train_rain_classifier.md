# Rain Classifier Training Documentation

## Overview

`train_rain_classifier.py` implements a binary rain classification model using MobileNetV4 transfer learning. The model learns to distinguish between images with rain and images without rain.

## Purpose

Train a deep learning model to automatically detect the presence of rain in images, particularly in cloud/weather imagery. Uses transfer learning from ImageNet-pretrained MobileNetV4 for efficient training with limited data.

## Architecture

### Base Model: MobileNetV4

- **Pretrained on**: ImageNet (1.2M images, 1000 classes)
- **Architecture**: MobileNetV4 with efficient conv blocks and optional hybrid attention
- **Input Size**: 224×224×3 (RGB)
- **Features**: Depthwise separable convolutions, universal inverted bottlenecks, progressive training
- **Source**: timm library (pytorch-image-models)
- **Paper**: https://arxiv.org/abs/2404.10518

### Transfer Learning Approach

```python
# Create MobileNetV4 with pre-trained weights
model = timm.create_model(
    'mobilenetv4_conv_medium.e500_r224_in1k',
    pretrained=True,
    num_classes=2  # Automatically replaces final layer for binary classification
)
```

**Why MobileNetV4?**
- Improved efficiency over MobileNetV3
- Better accuracy-to-latency tradeoff
- Latest architecture design (2024)
- Optimized for mobile/edge deployment
- Strong transfer learning performance
- Multiple variants available (small, medium, large, hybrid)

## Classes and Functions

### Class: RainDataset

Custom PyTorch Dataset for loading rain classification data.

**Constructor:**
```python
RainDataset(image_paths, labels, transform=None)
```

**Parameters:**
- `image_paths` (list): List of image file paths
- `labels` (list): List of labels (0=no rain, 1=rain)
- `transform` (callable): Optional transform pipeline

**Methods:**
- `__len__()`: Returns dataset size
- `__getitem__(idx)`: Returns (image, label) tuple at index

**Example:**
```python
train_dataset = RainDataset(
    image_paths=['img1.jpg', 'img2.jpg'],
    labels=[0, 1],
    transform=train_transform
)
```

### Function: prepare_dataset()

Automatically prepares dataset from folder structure.

**Signature:**
```python
prepare_dataset(rain_folder='overlayed_images', no_rain_folder='.')
```

**Parameters:**
- `rain_folder` (str): Folder containing rain images
- `no_rain_folder` (str): Folder containing no-rain images

**Returns:**
- `image_paths` (list): All image paths
- `labels` (list): Corresponding labels (0 or 1)

**Logic:**
1. Find all images in `rain_folder` → label as 1 (rain)
2. Find base images in `no_rain_folder` → label as 0 (no rain)
3. Exclude images already in rain folder or output folders
4. Return combined lists

**Example:**
```python
image_paths, labels = prepare_dataset(
    rain_folder='overlayed_images',
    no_rain_folder='.'
)
# Returns: [...], [1, 1, 1, 0, 0, ...]
```

### Function: create_model()

Creates and configures the MobileNetV4 model.

**Signature:**
```python
create_model(num_classes=2, pretrained=True)
```

**Parameters:**
- `num_classes` (int): Number of output classes (2 for binary)
- `pretrained` (bool): Load ImageNet weights

**Returns:**
- Modified MobileNetV4 model

**Implementation:**
```python
model = timm.create_model(
    'mobilenetv4_conv_medium.e500_r224_in1k',
    pretrained=pretrained,
    num_classes=num_classes  # Automatically replaces final layer
)
```

**Available MobileNetV4 Variants in timm:**
- `mobilenetv4_conv_small`: Smallest and fastest
- `mobilenetv4_conv_medium`: Good balance (default in this project)
- `mobilenetv4_conv_large`: Larger conv-only model
- `mobilenetv4_hybrid_medium`: Medium with hybrid (conv + attention) blocks
- `mobilenetv4_hybrid_large`: Largest with hybrid architecture

### Function: train_model()

Main training loop with validation.

**Signature:**
```python
train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=10,
    device='cuda'
)
```

**Parameters:**
- `model`: PyTorch model to train
- `train_loader`: Training DataLoader
- `val_loader`: Validation DataLoader
- `criterion`: Loss function (CrossEntropyLoss)
- `optimizer`: Optimizer (Adam)
- `num_epochs` (int): Number of training epochs
- `device` (str): 'cuda' or 'cpu'

**Returns:**
- `history` (dict): Training metrics
  - `train_loss`: List of training losses per epoch
  - `train_acc`: List of training accuracies per epoch
  - `val_loss`: List of validation losses per epoch
  - `val_acc`: List of validation accuracies per epoch

**Training Process:**
1. For each epoch:
   - **Training phase**: Forward pass, compute loss, backward pass, update weights
   - **Validation phase**: Evaluate on validation set without gradients
   - **Checkpoint**: Save model if validation accuracy improves
2. Return training history

### Function: plot_training_history()

Visualizes training and validation metrics.

**Signature:**
```python
plot_training_history(history, save_path='training_history.png')
```

**Parameters:**
- `history` (dict): Dictionary from `train_model()`
- `save_path` (str): Output file path

**Output:**
- Saves 2-panel plot with loss and accuracy curves

## Data Pipeline

### Data Transformations

**Training Transforms (with augmentation):**
```python
transforms.Compose([
    transforms.Resize((224, 224)),           # Resize to model input
    transforms.RandomHorizontalFlip(),       # 50% chance horizontal flip
    transforms.RandomRotation(10),           # ±10° rotation
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize(                     # ImageNet normalization
        [0.485, 0.456, 0.406],               # Mean (RGB)
        [0.229, 0.224, 0.225]                # Std (RGB)
    )
])
```

**Validation Transforms (no augmentation):**
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Data Split

- **Training**: 80% of data
- **Validation**: 20% of data
- **Stratified**: Maintains class balance in both splits
- **Random State**: 42 (reproducible splits)

## Training Configuration

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 16 | Images per batch |
| Learning Rate | 0.001 | Adam optimizer LR |
| Epochs | 10 | Training iterations |
| Optimizer | Adam | Adaptive learning rate |
| Loss Function | CrossEntropyLoss | For classification |
| Device | CUDA (if available) | GPU acceleration |

### Optimizer: Adam

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Why Adam?**
- Adaptive learning rates per parameter
- Works well with transfer learning
- Good default choice for most tasks
- No manual learning rate tuning needed

### Loss Function: CrossEntropyLoss

```python
criterion = nn.CrossEntropyLoss()
```

**Why CrossEntropyLoss?**
- Standard for multi-class classification
- Combines softmax and negative log-likelihood
- Numerically stable
- Works with class indices (0, 1)

## Usage

### Basic Training

```bash
python train_rain_classifier.py
```

**Process:**
1. Loads images from `overlayed_images/` (rain) and `.` (no rain)
2. Splits into 80/20 train/validation
3. Creates data loaders with augmentation
4. Trains MobileNetV4 (conv_medium variant) for 10 epochs
5. Saves best model as `best_rain_classifier.pth`
6. Generates `training_history.png` plot

### Custom Training

Modify parameters in the script:

```python
# Change epochs
num_epochs = 20

# Change batch size
batch_size = 32

# Change learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Use a different MobileNetV4 variant
model = timm.create_model(
    'mobilenetv4_conv_small.e500_r224_in1k',  # Smaller, faster
    pretrained=True,
    num_classes=2
)

# Use hybrid variant (slower but potentially better accuracy)
model = timm.create_model(
    'mobilenetv4_hybrid_large.e500_r224_in1k',
    pretrained=True,
    num_classes=2
)
```

## Output Files

### 1. best_rain_classifier.pth

**Content**: Model state dict (trained weights)

**Size**: ~10-15 MB (MobileNetV4 conv_medium)

**Loading:**
```python
model = timm.create_model(
    'mobilenetv4_conv_medium.e500_r224_in1k',
    pretrained=False,
    num_classes=2
)
model.load_state_dict(torch.load('best_rain_classifier.pth'))
```

### 2. training_history.png

**Content**: 2-panel plot
- Left: Training/validation loss curves
- Right: Training/validation accuracy curves

**Format**: PNG image

**Use**: Diagnose training (overfitting, convergence, etc.)

## Training Output

### Console Output

```
Rain Binary Classification with MobileNetV4
============================================================

Using device: cuda

Preparing dataset...
Found 16 rain images
Found 2 no-rain images
Total dataset size: 18 images

Train set: 14 images
Validation set: 4 images

Creating MobileNetV4 model...

Starting training...
============================================================

Epoch [1/10]
--------------------------------------------------
Batch [5/1], Loss: 0.6234, Acc: 58.33%
Train Loss: 0.5891, Train Acc: 64.29%
Val Loss: 0.4523, Val Acc: 75.00%
Saved best model with validation accuracy: 75.00%

...

============================================================
Training completed!
Best model saved as 'best_rain_classifier.pth'
Training history plot saved as 'training_history.png'
```

## Model Performance

### Expected Metrics

With sufficient diverse data:
- **Validation Accuracy**: 95-99%
- **Training Time**: 5-10 minutes (GPU), 30-60 minutes (CPU)
- **Inference Speed**: <50ms per image (GPU), ~200ms (CPU)

### Performance Factors

**Good Performance:**
- Balanced dataset (equal rain/no-rain samples)
- Diverse rain patterns
- Varied cloud types
- Sufficient training data (>100 images per class)

**Poor Performance:**
- Imbalanced data
- Limited rain variations
- Overfitting (high train acc, low val acc)
- Insufficient epochs

## Advanced Usage

### Fine-Tuning Specific Layers

```python
# Freeze early layers, only train classifier
for param in model.features.parameters():
    param.requires_grad = False

# Only classifier parameters will be updated
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# In training loop
for epoch in range(num_epochs):
    train_one_epoch()
    validate()
    scheduler.step()  # Decay LR every 5 epochs
```

### Early Stopping

```python
patience = 3
best_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    val_loss = validate()
    
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        save_model()
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

## Dependencies

- `torch`: PyTorch deep learning framework
- `torchvision`: Transforms and utilities
- `timm`: PyTorch Image Models (for MobileNetV4)
- `pillow`: Image loading
- `numpy`: Array operations
- `sklearn`: Train/test split
- `matplotlib`: Plotting

## Troubleshooting

### CUDA Out of Memory

**Solution:**
- Reduce batch size: `batch_size = 8`
- Use smaller model: `mobilenetv4_conv_small` (via timm)
- Use CPU: `device = 'cpu'`

### Low Validation Accuracy

**Solutions:**
- Increase training data
- Train for more epochs
- Check data balance
- Review augmentation
- Verify labels are correct

### Overfitting (High Train, Low Val Acc)

**Solutions:**
- Add more data augmentation
- Reduce model complexity
- Add dropout
- Use regularization
- Collect more training data

### Model Not Learning

**Solutions:**
- Check learning rate (try 0.0001 or 0.01)
- Verify data labels
- Check input normalization
- Increase epochs
- Try different optimizer

## Best Practices

1. **Data Quality**: Ensure correct labels and diverse samples
2. **Monitoring**: Watch training curves for overfitting
3. **Validation**: Always use separate validation set
4. **Checkpointing**: Save best model, not last epoch
5. **Reproducibility**: Set random seeds for consistent results
6. **Testing**: Evaluate on completely separate test set
