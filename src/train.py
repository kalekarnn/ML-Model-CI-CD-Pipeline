
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime

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

if __name__ == '__main__':
    train()
