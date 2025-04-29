# SmartConVulDetect
A Python-based Smart Contract Vulnerability Detection system that combines neural networks, interpretable graph features, and expert pattern fusion for enhanced security analysis.

## Overview

The system architecture consists of three main components:
1. Local Expert Pattern Extraction: Extracts vulnerability-specific patterns from the code
2. Graph Construction and Normalization: Transforms source code into a global semantic graph
3. Attentive Multi-Encoder Network: Combines expert patterns and graph features for vulnerability detection with explainable weights

## Required Packages
* Python 3 or above
* TensorFlow 2.0 or above
* scikit-learn 0.20.2
* NumPy 1.18 or above

### Installation
```shell
pip install --upgrade pip
pip install tensorflow==2.0
pip install scikit-learn==0.20.2
pip install numpy==1.18
```

## Dataset
The system evaluates smart contract vulnerabilities on two benchmark datasets:
1. Ethereum Smart Contract (ESC)
   - Focuses on reentrancy and timestamp dependence vulnerability

## Usage
Run the program using:
```shell
python3 VulDetector.py
```

You can customize parameters using command-line arguments:
```shell
python3 VulDetector.py --model EncoderWeight --lr 0.002 --dropout 0.2 --epochs 100 --batch_size 32
```

All configurable parameters can be found in `parser.py`.

## Case Study
Our system provides interpretable vulnerability detection through feature weight analysis. The following visualization shows how the system analyzes a real-world smart contract function for reentrancy vulnerability:


The visualization demonstrates:
- Global graph representation of the code
- Three local patterns for reentrancy detection
- Weight distribution between global and local features
- Clear explanation of the prediction reasoning

## Contact
For questions or support, please create an issue in the repository.