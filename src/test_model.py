import torch
import pytest
from torchvision import datasets, transforms
from model import MNISTModel
from torch import nn

def test_model_parameters_less_than_25000():
    model = MNISTModel()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"


def test_model_accuracy_greater_than_95():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    # Load the latest model
    import glob
    import os
    model_files = glob.glob('mnist_model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))

    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', 
                                train=False, 
                                transform=transform,
                                download=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=1000,
                                            shuffle=False)
    
    # Test accuracy
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 95, f"Accuracy is {accuracy}%, should be > 95%"


def test_model_output_shape():
    model = MNISTModel()
    batch_size = 32
    input_tensor = torch.randn(batch_size, 1, 28, 28)  # MNIST image size is 28x28
    output = model(input_tensor)
    assert output.shape == (batch_size, 10), f"Expected output shape (32, 10), got {output.shape}"

def test_model_forward_pass():
    model = MNISTModel()
    x = torch.randn(1, 1, 28, 28)
    try:
        output = model(x)
        assert True, "Forward pass successful"
    except Exception as e:
        assert False, f"Forward pass failed with error: {str(e)}"

def test_model_layers():
    model = MNISTModel()
    # Test conv layers
    assert isinstance(model.conv1, nn.Conv2d), "First layer should be Conv2d"
    assert isinstance(model.conv2, nn.Conv2d), "Second layer should be Conv2d"
    # Test fully connected layers
    assert isinstance(model.fc1, nn.Linear), "Third layer should be Linear"
    assert isinstance(model.fc2, nn.Linear), "Fourth layer should be Linear"
    # Test activation and pooling
    assert isinstance(model.pool, nn.MaxPool2d), "Should have MaxPool2d layer"
    assert isinstance(model.relu, nn.ReLU), "Should have ReLU activation"
