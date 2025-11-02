# MNIST Handwritten Digit Classifier

A deep neural network implementation for classifying handwritten digits from the MNIST dataset using PyTorch, achieving 97%+ accuracy on test data.

## Overview

This project implements a 2-layer fully connected neural network that classifies 28×28 grayscale images of handwritten digits (0-9). The model processes 60,000 training images and achieves high accuracy on 10,000 test images through optimized batch processing and GPU acceleration.

## Features

- **Deep Learning Architecture**: 2-layer neural network (784 → 128 → 10 neurons)
- **GPU Acceleration**: Automatic device detection for CUDA, MPS (Apple Silicon), and CPU
- **Data Preprocessing**: Image normalization using MNIST-specific statistics (mean: 0.1307, std: 0.3081)
- **Batch Processing**: Efficient DataLoader implementation with batch size of 64
- **Visualization Tools**:
  - Pixel-level image display with value overlays
  - Real-time training metrics (loss and accuracy curves)
  - Model prediction visualization with color-coded accuracy indicators

## Technologies Used

- **Python 3.x**
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **torchvision**: Dataset handling and transforms
- **PIL**: Image processing

## Model Architecture

```
Input Layer:  784 neurons (28×28 flattened image)
Hidden Layer: 128 neurons (ReLU activation)
Output Layer: 10 neurons (digit classes 0-9)
```

**Training Configuration**:
- Loss Function: CrossEntropyLoss
- Optimizer: Adam (learning rate: 0.001)
- Epochs: 5
- Batch Size: 64 (training), 1000 (testing)

## Results

- **Test Accuracy**: 97%+ on 10,000 test images
- **Training Loss**: Reduced from ~2.3 to <0.2 over 5 epochs
- **Processing Speed**: GPU-accelerated training with automatic device selection

## Files

- `5.py`: Main training and evaluation script
- `helper_utils.py`: Visualization utilities for images, predictions, and metrics

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ml-projects
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib pillow
```

## Usage

Run the classifier:
```bash
python 5.py
```

The script will automatically:
- Download the MNIST dataset (if not present)
- Train the model for 5 epochs
- Display training progress and test accuracy
- Visualize sample predictions

## Visualizations

The project includes three types of visualizations:
1. **Image Display**: Shows individual images with pixel values
2. **Prediction Grid**: Displays model predictions for sample digits from each class (color-coded: green for correct, red for incorrect)
3. **Training Metrics**: Plots training loss and test accuracy over epochs

## Project Structure

```
ml-projects/
├── 5.py                  # Main training script
├── helper_utils.py       # Visualization utilities
├── data/                 # MNIST dataset (auto-downloaded)
├── venv/                 # Virtual environment
└── README.md            # Project documentation
```

## License

MIT License

## Author

Jaskirat Singh Sohal
