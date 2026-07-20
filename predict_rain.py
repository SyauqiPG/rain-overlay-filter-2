"""
Inference script for rain classification using trained MobileNetV4 model.
"""

import torch
from torchvision import transforms
import timm
from PIL import Image
import argparse
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    ort = None


MODEL_NAME = 'mobilenetv4_conv_medium.e500_r224_in1k'
INPUT_SIZE = 224
NUM_CLASSES = 2
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ['No Rain', 'Rain']


class ONNXRainModel:
    """Light wrapper for ONNX Runtime inference context."""

    def __init__(self, session, runtime_device):
        self.session = session
        self.runtime_device = runtime_device
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name


def _resolve_onnx_providers(requested_device='cuda'):
    """Return ONNX Runtime providers with safe fallback."""
    available = ort.get_available_providers()
    if requested_device == 'cuda' and 'CUDAExecutionProvider' in available:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'cuda'

    if requested_device == 'cuda':
        print('CUDAExecutionProvider not available, falling back to CPUExecutionProvider.')

    return ['CPUExecutionProvider'], 'cpu'


def _softmax_numpy(logits):
    """Numerically stable softmax for ONNX logits."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def load_model(model_path, device='cuda'):
    """
    Load the trained model.
    
    Args:
        model_path: Path to saved model weights
        device: Device to load model on
    
    Returns:
        Loaded model ready for inference
    """
    model_extension = Path(model_path).suffix.lower()

    if model_extension == '.onnx':
        if ort is None:
            raise ImportError('onnxruntime is required to load .onnx models. Install with: pip install onnxruntime')

        providers, runtime_device = _resolve_onnx_providers(device)
        session = ort.InferenceSession(model_path, providers=providers)
        print(f'Using ONNX Runtime providers: {session.get_providers()}')
        return ONNXRainModel(session=session, runtime_device=runtime_device)

    resolved_device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

    # Create model architecture (same as training) using timm
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=resolved_device))
    model = model.to(resolved_device)
    model.eval()

    return model


def predict_image(model, image_path, device='cuda'):
    """
    Predict whether an image contains rain.
    
    Args:
        model: Trained model
        image_path: Path to image
        device: Device to run inference on
    
    Returns:
        Prediction (0=no rain, 1=rain) and confidence
    """
    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    if isinstance(model, ONNXRainModel):
        logits = model.session.run(
            [model.output_name],
            {model.input_name: image_tensor.numpy().astype(np.float32)}
        )[0]
        probabilities = _softmax_numpy(logits)
        prediction = int(np.argmax(probabilities, axis=1)[0])
        confidence_score = float(np.max(probabilities, axis=1)[0] * 100)
        return prediction, confidence_score, CLASS_NAMES[prediction]

    resolved_device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    image_tensor = image_tensor.to(resolved_device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    prediction = predicted.item()
    confidence_score = confidence.item() * 100

    return prediction, confidence_score, CLASS_NAMES[prediction]


def main():
    parser = argparse.ArgumentParser(description='Predict rain in images')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='best_rain_classifier.pth',
                       help='Path to trained model (.pth or .onnx)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, args.device)

    if isinstance(model, ONNXRainModel):
        print(f"Using ONNX runtime device: {model.runtime_device}")
        prediction_device = model.runtime_device
    else:
        prediction_device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        print(f"Using device: {prediction_device}")
    
    # Make prediction
    print(f"Analyzing image: {args.image}")
    prediction, confidence, class_name = predict_image(model, args.image, prediction_device)
    
    print("\n" + "=" * 50)
    print(f"Prediction: {class_name}") #test
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    main()