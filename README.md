# Smart Contract Vulnerability Detection System

A comprehensive system for detecting vulnerabilities in smart contracts using a hybrid approach combining pattern-based and graph-based features with deep learning.

## Overview

The system implements a sophisticated vulnerability detection framework that combines:
- Pattern-based feature extraction
- Graph-based semantic analysis
- Deep learning models for vulnerability classification

## Features

- **Hybrid Detection Approach**: Combines pattern-based and graph-based features for comprehensive vulnerability detection
- **Multiple Vulnerability Types**: Supports detection of various smart contract vulnerabilities
- **Web Interface**: User-friendly web application for easy interaction
- **Robust Processing**: Advanced preprocessing and feature extraction
- **Model Persistence**: Save and load trained models for future use

## Project Structure

```
.
├── VulDetector.py          # Main vulnerability detection system
├── model.py               # Neural network model implementation
├── preprocessing.py       # Data preprocessing and feature extraction
├── web_application.py     # Web interface for user interaction
├── parser.py             # Configuration and parameter handling
├── pattern_feature/      # Pattern-based features
├── graph_feature/        # Graph-based features
├── models/              # Trained model storage
├── results/            # Detection results
├── logs/              # System logs
└── data_example/     # Example smart contracts
```

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- scikit-learn
- Other dependencies listed in requirements.txt

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Training the Model**:
```bash
python VulDetector.py --vulnerability_type <type> --epochs <num_epochs>
```

2. **Web Interface**:
```bash
python web_application.py
```

3. **Command Line Options**:
- `--vulnerability_type`: Type of vulnerability to detect
- `--epochs`: Number of training epochs
- `--save_model`: Whether to save the trained model
- See `parser.py` for full list of options

## Model Architecture

The system uses a hybrid neural network architecture:
- Graph Neural Network (GNN) layers for processing structural information
- Pattern feature processing for local vulnerability patterns
- Attention mechanisms for feature importance
- Custom layers for specialized processing

## Data Processing

The system processes smart contracts through:
1. Pattern feature extraction
2. Graph-based semantic analysis
3. Feature normalization and augmentation
4. Hybrid feature combination

## Contributing

Feel free to submit issues and enhancement requests.
