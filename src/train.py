import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import matplotlib.pyplot as plt

def train():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', 
                                 train=True, 
                                 transform=transform,
                                 download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=64,
                                             shuffle=True)

    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    total_step = len(train_loader)
    for epoch in range(1):  # Just 1 epoch as requested
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/1], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

    # Save the model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f'mnist_model_{timestamp}.pth')

def show_augmented_images(original_image, augmented_images, save_path='augmentation_examples.png'):
    """Display original and augmented images"""
    plt.figure(figsize=(15, 3))
    
    # Show original
    plt.subplot(1, 5, 1)
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Show augmented versions
    for idx, img in enumerate(augmented_images):
        plt.subplot(1, 5, idx + 2)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f'Augmented {idx + 1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def image_augment():
    # Basic transform for visualization
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create and save augmentation examples
    example_dataset = datasets.MNIST('./data', train=True, download=True, transform=basic_transform)
    example_image = example_dataset[0][0]

    # Generate augmented examples
    augmented_images = []
    for _ in range(4):
        aug_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
        ])
        augmented_images.append(aug_transform(example_image))
    
    # Save augmentation examples
    show_augmented_images(example_image, augmented_images)
    

if __name__ == '__main__':
    train()
    image_augment()
