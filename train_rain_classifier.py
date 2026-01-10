"""
Rain Binary Classification using MobileNetV3 Transfer Learning
Based on: https://medium.com/@RobuRishabh/understanding-and-implementing-mobilenetv3-422bd0bdfb5a
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class RainDataset(Dataset):
    """Custom dataset for rain/no-rain binary classification."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def prepare_dataset(rain_folder='overlayed_images', no_rain_folder='.'):
    """
    Prepare dataset with rain and no-rain images.
    
    Args:
        rain_folder: Folder containing images with rain overlay
        no_rain_folder: Folder containing original images (no rain)
    
    Returns:
        Lists of image paths and corresponding labels
    """
    image_paths = []
    labels = []
    
    # Get rain images (label = 1)
    rain_images = list(Path(rain_folder).glob('*.jpg')) + list(Path(rain_folder).glob('*.png'))
    for img_path in rain_images:
        image_paths.append(str(img_path))
        labels.append(1)  # Rain
    
    print(f"Found {len(rain_images)} rain images")
    
    # Get no-rain images (label = 0)
    # Look for cumulonimbus images in the no_rain_folder
    no_rain_images = []
    for pattern in ['6_cumulonimbus*.jpg', '*.jpg', '*.png']:
        found = list(Path(no_rain_folder).glob(pattern))
        # Filter out rain overlay images
        found = [f for f in found if 'overlayed' not in str(f) and 'output' not in str(f)]
        no_rain_images.extend(found)
    
    # Remove duplicates
    no_rain_images = list(set(no_rain_images))
    
    for img_path in no_rain_images:
        image_paths.append(str(img_path))
        labels.append(0)  # No rain
    
    print(f"Found {len(no_rain_images)} no-rain images")
    print(f"Total dataset size: {len(image_paths)} images")
    
    return image_paths, labels


def create_model(num_classes=2, pretrained=True):
    """
    Create MobileNetV3-Large model for binary classification.
    
    Args:
        num_classes: Number of output classes (2 for binary)
        pretrained: Whether to use pretrained ImageNet weights
    
    Returns:
        Modified MobileNetV3 model
    """
    # Load pretrained MobileNetV3-Large
    model = models.mobilenet_v3_large(pretrained=pretrained)
    
    # Modify the final classification layer
    # MobileNetV3 classifier structure: [Linear(960, 1280), Hardswish, Dropout, Linear(1280, 1000)]
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features=in_features, out_features=num_classes)
    
    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=10, device='cuda'):
    """
    Train the model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        Training history (losses and accuracies)
    """
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print('-' * 50)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f'Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100 * correct / total:.2f}%')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_rain_classifier.pth')
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
    
    return history


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'\nSaved training history plot to {save_path}')


def main():
    """Main training pipeline."""
    
    print("Rain Binary Classification with MobileNetV3")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    image_paths, labels = prepare_dataset(
        rain_folder='overlayed_images',
        no_rain_folder='./no-rain'
    )
    
    # Split dataset into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTrain set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    
    # Define data transformations (following ImageNet normalization)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = RainDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = RainDataset(val_paths, val_labels, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Create model
    print("\nCreating MobileNetV3-Large model...")
    model = create_model(num_classes=2, pretrained=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("\nStarting training...")
    print("=" * 60)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device=device
    )
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved as 'best_rain_classifier.pth'")
    print(f"Training history plot saved as 'training_history.png'")


if __name__ == '__main__':
    main()
