import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

class COCOClassificationDataset(Dataset):
    """Convert COCO object detection annotations to image classification dataset"""
    
    def __init__(self, annotation_file, image_dir, transform=None, multi_label=False):
        """
        Args:
            annotation_file: Path to _annotations.coco.json
            image_dir: Directory containing images
            transform: PyTorch transforms
            multi_label: If True, handles images with multiple classes
        """
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.multi_label = multi_label
        
        # Create category mapping (exclude supercategory)
        self.categories = {cat['id']: cat['name'] 
                          for cat in self.coco_data['categories'] 
                          if cat['name'] != 'dashcam-9Zyu'}
        
        self.class_to_idx = {name: idx for idx, name in enumerate(sorted(self.categories.values()))}
        self.num_classes = len(self.class_to_idx)
        
        # Map images to their categories
        self.image_labels = self._create_image_label_mapping()
        
        # Create list of valid samples
        self.samples = [(img_id, labels) for img_id, labels in self.image_labels.items() if labels]
        
        print(f"Dataset initialized:")
        print(f"  Total images: {len(self.samples)}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Classes: {list(self.class_to_idx.keys())}")
        
    def _create_image_label_mapping(self):
        """Create mapping from image_id to category labels"""
        image_labels = defaultdict(set)
        
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            
            if cat_id in self.categories:
                cat_name = self.categories[cat_id]
                image_labels[img_id].add(cat_name)
        
        return image_labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_id, label_names = self.samples[idx]
        
        # Get image info
        img_info = next(img for img in self.coco_data['images'] if img['id'] == img_id)
        img_path = self.image_dir / img_info['file_name']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert labels to indices
        if self.multi_label:
            # Multi-label classification (one-hot encoding)
            label = torch.zeros(self.num_classes)
            for name in label_names:
                label[self.class_to_idx[name]] = 1
        else:
            # Single-label classification (use first label or most common)
            label = self.class_to_idx[list(label_names)[0]]
        
        return image, label


def get_transforms(image_size=224, augment=True):
    """Get training and validation transforms"""
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_resnet_model(num_classes, model_name='resnet50', pretrained=True):
    """
    Create ResNet model for classification
    
    Args:
        num_classes: Number of output classes
        model_name: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
        pretrained: Use ImageNet pretrained weights
    """
    # Load pretrained ResNet
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    if scheduler:
        scheduler.step()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_model(
    train_annotation_file,
    train_image_dir,
    val_annotation_file=None,
    val_image_dir=None,
    model_name='resnet50',
    num_epochs=50,
    batch_size=32,
    learning_rate=0.001,
    weight_decay=1e-4,
    image_size=224,
    pretrained=True,
    save_dir='checkpoints',
    device=None
):
    """
    Main training function
    
    Args:
        train_annotation_file: Path to training annotations
        train_image_dir: Path to training images
        val_annotation_file: Path to validation annotations (optional)
        val_image_dir: Path to validation images (optional)
        model_name: ResNet variant to use
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        image_size: Input image size
        pretrained: Use ImageNet pretrained weights
        save_dir: Directory to save checkpoints
        device: Device to train on (cuda/cpu)
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get transforms
    train_transform, val_transform = get_transforms(image_size, augment=True)
    
    # Create datasets
    train_dataset = COCOClassificationDataset(
        train_annotation_file, 
        train_image_dir, 
        transform=train_transform
    )
    
    num_classes = train_dataset.num_classes
    class_names = list(train_dataset.class_to_idx.keys())
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # Validation dataset (if provided)
    if val_annotation_file and val_image_dir:
        val_dataset = COCOClassificationDataset(
            val_annotation_file, 
            val_image_dir, 
            transform=val_transform
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
    else:
        val_loader = None
    
    # Create model
    print(f"\nCreating {model_name} model with {num_classes} classes...")
    model = create_resnet_model(num_classes, model_name, pretrained)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Classes: {class_names}\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # Validate
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_names': class_names
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'class_names': class_names
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        print()
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': class_names,
        'history': history
    }, os.path.join(save_dir, 'final_model.pth'))
    
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, history


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'train_annotation_file': 'train/_annotations.coco.json',
        'train_image_dir': 'train/',
        'val_annotation_file': 'valid/_annotations.coco.json',  # Optional
        'val_image_dir': 'valid/',  # Optional
        'model_name': 'resnet50',  # Options: resnet18, resnet34, resnet50, resnet101, resnet152
        'num_epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'image_size': 224,
        'pretrained': True,
        'save_dir': 'checkpoints'
    }
    
    # Train model
    model, history = train_model(**config)
    
    # Plot training history (optional)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        if history['val_loss']:
            plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        if history['val_acc']:
            plt.plot(history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history plot saved as 'training_history.png'")
    except ImportError:
        print("Matplotlib not available for plotting")