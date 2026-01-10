"""
Inference script for rain classification using trained MobileNetV3 model.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse


def load_model(model_path, device='cuda'):
    """
    Load the trained model.
    
    Args:
        model_path: Path to saved model weights
        device: Device to load model on
    
    Returns:
        Loaded model ready for inference
    """
    # Create model architecture (same as training)
    model = models.mobilenet_v3_large(pretrained=False)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features=in_features, out_features=2)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_names = ['No Rain', 'Rain']
    prediction = predicted.item()
    confidence_score = confidence.item() * 100
    
    return prediction, confidence_score, class_names[prediction] # return


def main():
    parser = argparse.ArgumentParser(description='Predict rain in images')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='best_rain_classifier.pth',
                       help='Path to trained model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device availability
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)
    
    # Make prediction
    print(f"Analyzing image: {args.image}")
    prediction, confidence, class_name = predict_image(model, args.image, device)
    
    print("\n" + "=" * 50)
    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    main()
