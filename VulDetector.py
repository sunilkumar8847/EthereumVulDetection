#!/usr/bin/env python
"""
Main entry point for AMEVulDetector.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from model import build_model, process_graph_feature, process_pattern_feature
from preprocessing import get_graph_feature, get_pattern_feature
import logging
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_vulnerability_model(vulnerability_type, epochs=20, save_model=False):
    """
    Train a model for a specific vulnerability type.
    
    Args:
        vulnerability_type (str): Type of vulnerability to train for
        epochs (int): Number of epochs to train for
        save_model (bool): Whether to save the trained model
    """
    try:
        logging.info(f"Training model for {vulnerability_type} vulnerability...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Load graph features
        train_graph, test_graph = get_graph_feature(vulnerability_type)
        train_size = len(train_graph)
        test_size = len(test_graph)
        
        logging.info(f"Graph features loaded - Train: {train_size} samples, Test: {test_size} samples")
        
        # Load pattern features
        train_pattern, test_pattern = get_pattern_feature(vulnerability_type, train_size, test_size)
        if train_pattern is None or test_pattern is None:
            raise ValueError("Failed to load pattern features")
            
        logging.info(f"Pattern features loaded - Train: {train_pattern.shape}, Test: {test_pattern.shape}")
        
        # Create balanced labels
        train_labels = np.zeros(train_size)
        train_labels[:train_size//2] = 1  # First half are vulnerable
        
        test_labels = np.zeros(test_size)
        test_labels[:test_size//2] = 1  # First half are vulnerable
        
        # Shuffle the data and labels together
        train_indices = np.random.permutation(train_size)
        test_indices = np.random.permutation(test_size)
        
        train_pattern = train_pattern[train_indices]
        train_graph = train_graph[train_indices]
        train_labels = train_labels[train_indices]
        
        test_pattern = test_pattern[test_indices]
        test_graph = test_graph[test_indices]
        test_labels = test_labels[test_indices]
        
        logging.info("Building model...")
        model = build_model()
        model.summary()
        
        # Use validation split only if we have enough samples
        use_validation_split = train_size >= 100  # Increased threshold for better stability
        validation_split = 0.2 if use_validation_split else 0.0
        
        # Adjust batch size to not exceed number of samples
        batch_size = min(32, train_size)
        
        # Create callbacks
        callbacks = []
        if use_validation_split:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ))
        
        logging.info("Starting training...")
        history = model.fit(
            [train_graph, train_pattern],
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split if use_validation_split else 0.0,
            callbacks=callbacks if callbacks else None,
            verbose=1
        )
        
        # Evaluate model
        logging.info("Evaluating model...")
        metrics = model.evaluate(
            [test_graph, test_pattern],
            test_labels,
            verbose=0,
            return_dict=True
        )
        
        for metric_name, value in metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")
        
        if save_model:
            model_path = f"models/{vulnerability_type}_model.h5"
            model.save(model_path)
            logging.info(f"Model saved to {model_path}")
            
    except Exception as e:
        logging.error(f"Failed to train model for {vulnerability_type}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train vulnerability detection models')
    parser.add_argument('-v', '--vulnerabilities', type=str, required=True,
                      help='Comma-separated list of vulnerabilities to train (e.g., "reentrancy,timestamp")')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of training epochs')
    parser.add_argument('--save_model', action='store_true',
                      help='Save the trained model')
    
    args = parser.parse_args()
    
    # Split vulnerabilities
    vulnerabilities = [v.strip() for v in args.vulnerabilities.split(',')]
    
    # Train models for each vulnerability
    for vul_type in vulnerabilities:
        try:
            train_vulnerability_model(
                vulnerability_type=vul_type,
                epochs=args.epochs,
                save_model=args.save_model
            )
        except Exception as e:
            logging.error(f"Failed to train model for {vul_type}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
