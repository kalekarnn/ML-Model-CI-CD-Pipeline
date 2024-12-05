# ML Model CI/CD Pipeline

This repository contains a machine learning project with automated CI/CD pipeline for training, testing and deploying a CNN model on the MNIST dataset.

## Project Structure

# .
# ├── .github/
# │   └── workflows/
# │       └── ml_pipeline.yml
# ├── src/
# │   ├── model.py
# │   ├── train.py
# │   └── test_model.py
# ├── requirements.txt
# └── README.md

## Model Details

- **Input**: 28x28 grayscale images
- **Convolutional Layers**: 2 layers with increasing filters (32, 64)
- **Pooling**: Max pooling to reduce spatial dimensions
- **Dense Layers**: 2 fully connected layers
- **Regularization**: Dropout layer with 0.5 rate to prevent overfitting
- **Output**: 10 classes (digits 0-9)

## Key Features

- Automated training and testing pipeline
- Dataset size limited to 25,000 samples for faster training
- Model parameter count kept under 25,000 for efficiency
- Automated accuracy testing (>95% required)

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```


3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Local Development

1. Train the model:
```bash
python src/train.py
```
This will:
- Load 25,000 random samples from MNIST
- Train the CNN model
- Save the model with timestamp and device info

2. Run tests:
```bash
python -m pytest src/test_model.py -v
```
Tests verify:
- Model architecture (input/output shapes)
- Parameter count (<25,000)
- Model accuracy (>95% on test set)

## CI/CD Pipeline

The GitHub Actions workflow (`ml-pipeline.yml`) automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains model
4. Runs tests
5. Saves model artifact

The pipeline runs on every push to the repository.


## Model Artifacts

Trained models are saved with the naming convention:
```
mnist_model_{timestamp}.pth
```

These files are:
- Ignored by git (.gitignore)
- Uploaded as artifacts in GitHub Actions
- Used by test cases to verify model performance


## Testing

The test suite (`test_model.py`) includes:
- Architecture tests
  - Verifies input/output dimensions
  - Checks parameter count (<25,000)
- Performance tests
  - Loads latest trained model
  - Verifies accuracy >95% on MNIST test set

## Notes

- Training uses CPU by default (GPU if available)
- Model architecture is optimized for size while maintaining accuracy
- Test dataset is downloaded automatically when needed
- All trained models and datasets are excluded from git tracking