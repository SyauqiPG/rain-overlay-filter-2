# Rain Prediction Documentation

## Overview

`predict_rain.py` is an inference script for making predictions using a trained rain classification model. It loads a saved MobileNetV3 model and classifies whether an image contains rain.

## Purpose

Deploy the trained rain classification model for:
- Single image predictions
- Testing model performance
- Real-time rain detection
- Integration into other applications

## Functions

### load_model()

Loads a trained model from saved weights.

**Signature:**
```python
load_model(model_path, device='cuda')
```

**Parameters:**
- `model_path` (str): Path to saved model weights (.pth file)
- `device` (str): Device to load model on ('cuda' or 'cpu')

**Returns:**
- Loaded PyTorch model in evaluation mode

**Implementation:**
```python
# Recreate model architecture
model = models.mobilenet_v3_large(pretrained=False)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 2)

# Load trained weights
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()  # Set to evaluation mode
```

**Example:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('best_rain_classifier.pth', device)
```

### predict_image()

Predicts rain presence in a single image.

**Signature:**
```python
predict_image(model, image_path, device='cuda')
```

**Parameters:**
- `model`: Loaded PyTorch model
- `image_path` (str): Path to image file
- `device` (str): Device for inference

**Returns:**
- `prediction` (int): Class index (0=no rain, 1=rain)
- `confidence` (float): Confidence score (0-100%)
- `class_name` (str): Human-readable class name

**Processing Pipeline:**
1. Load image from path
2. Convert to RGB
3. Apply transforms (resize, normalize)
4. Add batch dimension
5. Forward pass through model
6. Apply softmax to get probabilities
7. Return prediction and confidence

**Example:**
```python
prediction, confidence, class_name = predict_image(
    model,
    'test_image.jpg',
    device
)

print(f"Prediction: {class_name} ({confidence:.2f}%)")
# Output: "Prediction: Rain (98.45%)"
```

### main()

Command-line interface for predictions.

**Usage:**
```bash
python predict_rain.py --image PATH [--model PATH] [--device DEVICE]
```

**Arguments:**
- `--image` (required): Path to image for prediction
- `--model` (optional): Path to model weights (default: 'best_rain_classifier.pth')
- `--device` (optional): Device to use - 'cuda' or 'cpu' (default: 'cuda')

**Example:**
```bash
python predict_rain.py --image overlayed_images/rainy_cloud.jpg
```

## Image Preprocessing

### Transform Pipeline

The same preprocessing used during training:

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Resize to 224×224
    transforms.ToTensor(),                   # Convert to tensor [0, 1]
    transforms.Normalize(                     # ImageNet normalization
        [0.485, 0.456, 0.406],               # Mean per channel
        [0.229, 0.224, 0.225]                # Std per channel
    )
])
```

**Why This Matters:**
- Must match training preprocessing exactly
- ImageNet normalization required for pretrained models
- Incorrect preprocessing → poor predictions

### Input Requirements

- **Format**: Any PIL-supported format (JPG, PNG, BMP, etc.)
- **Size**: Any size (automatically resized to 224×224)
- **Channels**: RGB (grayscale images converted automatically)
- **Color Space**: sRGB

## Output Interpretation

### Class Labels

| Index | Class Name | Description |
|-------|-----------|-------------|
| 0 | No Rain | Clean image without rain |
| 1 | Rain | Image contains rain |

### Confidence Score

**Range**: 0-100%

**Interpretation:**
- **90-100%**: Very confident (highly certain)
- **70-90%**: Confident (reliable prediction)
- **50-70%**: Uncertain (borderline case)
- **<50%**: Very uncertain (should not happen with softmax max)

**Example Output:**
```
==================================================
Prediction: Rain
Confidence: 98.45%
==================================================
```

**Meaning**: Model is 98.45% confident the image contains rain.

## Usage Examples

### Command-Line Usage

**Basic prediction:**
```bash
python predict_rain.py --image test_image.jpg
```

**Custom model path:**
```bash
python predict_rain.py --image test.jpg --model models/rain_v2.pth
```

**Force CPU (no GPU):**
```bash
python predict_rain.py --image test.jpg --device cpu
```

**Full example:**
```bash
python predict_rain.py \
    --image overlayed_images/6_cumulonimbus_000005_heavy_rain_topdown_224x224.jpg \
    --model best_rain_classifier.pth \
    --device cuda
```

### Programmatic Usage

**Single prediction:**
```python
from predict_rain import load_model, predict_image
import torch

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('best_rain_classifier.pth', device)

# Predict
prediction, confidence, class_name = predict_image(
    model,
    'my_image.jpg',
    device
)

print(f"{class_name}: {confidence:.2f}%")
```

**Batch predictions:**
```python
from predict_rain import load_model, predict_image
import torch
from pathlib import Path

device = torch.device('cuda')
model = load_model('best_rain_classifier.pth', device)

# Predict on multiple images
image_paths = list(Path('test_images').glob('*.jpg'))

for img_path in image_paths:
    pred, conf, name = predict_image(model, str(img_path), device)
    print(f"{img_path.name}: {name} ({conf:.1f}%)")
```

**Integration example:**
```python
import torch
from predict_rain import load_model, predict_image
from PIL import Image

class RainDetector:
    def __init__(self, model_path='best_rain_classifier.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(model_path, self.device)
    
    def has_rain(self, image_path, threshold=0.5):
        """Returns True if rain is detected with confidence > threshold."""
        pred, conf, _ = predict_image(self.model, image_path, self.device)
        return pred == 1 and conf > threshold * 100

# Usage
detector = RainDetector()
if detector.has_rain('photo.jpg', threshold=0.7):
    print("Rain detected!")
```

## Performance

### Inference Speed

**GPU (CUDA):**
- First prediction: ~200-500ms (model loading)
- Subsequent: ~10-50ms per image
- Batch (optimal): ~5-20ms per image

**CPU:**
- First prediction: ~1-2 seconds
- Subsequent: ~100-300ms per image

**Optimization Tips:**
- Load model once, predict many times
- Use GPU when available
- Batch predictions for multiple images
- Use TorchScript for production deployment

### Memory Usage

**Model Size:**
- RAM: ~60 MB (MobileNetV3-Large)
- VRAM (GPU): ~100 MB including overhead

**Per Image:**
- Input image: <10 MB
- Processed tensor: ~0.6 MB (224×224×3 float32)

## Advanced Usage

### Batch Prediction

```python
import torch
from torchvision import transforms
from PIL import Image

def predict_batch(model, image_paths, device, batch_size=32):
    """Predict on multiple images efficiently."""
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    results = []
    
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load and transform batch
            images = []
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                images.append(transform(img))
            
            # Stack into batch tensor
            batch = torch.stack(images).to(device)
            
            # Predict
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, 1)
            
            # Store results
            for path, pred, conf in zip(batch_paths, predictions, confidences):
                results.append({
                    'path': path,
                    'prediction': pred.item(),
                    'confidence': conf.item() * 100
                })
    
    return results
```

### Probability Distribution

```python
def get_probabilities(model, image_path, device):
    """Get probability for each class."""
    from predict_rain import predict_image
    import torch
    from torchvision import transforms
    from PIL import Image
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()
    
    return {
        'no_rain': probs[0].item() * 100,
        'rain': probs[1].item() * 100
    }

# Usage
probs = get_probabilities(model, 'image.jpg', device)
print(f"No Rain: {probs['no_rain']:.2f}%")
print(f"Rain: {probs['rain']:.2f}%")
```

### Model Export (TorchScript)

```python
# Export model for production
model = load_model('best_rain_classifier.pth', 'cpu')
model.eval()

example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('rain_classifier_traced.pt')

# Load and use traced model
loaded_traced = torch.jit.load('rain_classifier_traced.pt')
# Use same as regular model
```

## Error Handling

### Common Issues

**Model file not found:**
```python
try:
    model = load_model('best_rain_classifier.pth', device)
except FileNotFoundError:
    print("Model file not found. Train model first using train_rain_classifier.py")
```

**Image file not found:**
```python
from pathlib import Path

image_path = 'test.jpg'
if not Path(image_path).exists():
    print(f"Image not found: {image_path}")
    exit(1)
```

**CUDA not available:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print("Warning: CUDA not available, using CPU (slower)")
```

## Dependencies

- `torch`: PyTorch for model loading and inference
- `torchvision`: Image transforms and models
- `pillow`: Image loading
- `argparse`: Command-line argument parsing

## Integration Examples

### Web API (Flask)

```python
from flask import Flask, request, jsonify
from predict_rain import load_model, predict_image
import torch

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('best_rain_classifier.pth', device)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Save uploaded image
    image = request.files['image']
    image.save('temp.jpg')
    
    # Predict
    pred, conf, name = predict_image(model, 'temp.jpg', device)
    
    return jsonify({
        'prediction': name,
        'confidence': conf,
        'class_index': pred
    })

if __name__ == '__main__':
    app.run(port=5000)
```

### Gradio Interface

```python
import gradio as gr
from predict_rain import load_model, predict_image
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('best_rain_classifier.pth', device)

def classify_image(image):
    # Gradio passes PIL Image directly
    image.save('temp.jpg')
    pred, conf, name = predict_image(model, 'temp.jpg', device)
    return f"{name}: {conf:.2f}%"

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Rain Classifier",
    description="Upload an image to detect rain"
)

interface.launch()
```

## Best Practices

1. **Load Once**: Load model once at startup, not per prediction
2. **Device Check**: Always check CUDA availability
3. **Error Handling**: Validate inputs before prediction
4. **Batch Processing**: Use batching for multiple images
5. **Monitoring**: Log predictions and confidence scores
6. **Thresholds**: Set confidence thresholds for critical applications
