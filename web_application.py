import os
import numpy as np
import tensorflow as tf
from models.EncoderWeight import EncoderWeight

class VulnerabilityDetector:
    """
    Class for detecting vulnerabilities in smart contracts
    """
    def __init__(self):
        # Create a dictionary to store models for different vulnerability types
        self.models = {}
        self.vulnerability_types = ['reentrancy', 'timestamp']
        
        # Load models if they exist
        for vul_type in self.vulnerability_types:
            model_path = f"models/saved/{vul_type}/EncoderWeight.weights.h5"
            if os.path.exists(model_path):
                # Initialize a model with empty placeholders
                model = EncoderWeight([], [], [], [], [], [], [], [], [], [])
                # Set the vulnerability type information
                model.vulnerability_type = vul_type
                model.prefix = vul_type[:2]
                model.result_dir = f"results/{vul_type}"
                
                # We need to build the model first before loading weights
                # This is just to initialize the architecture
                input_shape = (1, 1, 250)
                dummy_input1 = np.zeros(input_shape)
                dummy_input2 = np.zeros(input_shape)
                dummy_input3 = np.zeros(input_shape)
                dummy_input4 = np.zeros(input_shape)
                model.model.predict([dummy_input1, dummy_input2, dummy_input3, dummy_input4])
                
                # Now load the weights
                model.model.load_weights(model_path)
                
                # Add to our models dictionary
                self.models[vul_type] = model
                print(f"Loaded model for {vul_type} vulnerability detection")
            else:
                print(f"No model found for {vul_type} vulnerability detection at {model_path}")
    
    def detect_vulnerabilities(self, graph_features, pattern_features):
        """
        Detect vulnerabilities in the provided features
        
        Args:
            graph_features (numpy.ndarray): Graph features of shape (1, 1, 250)
            pattern_features (list): List of three pattern features, each of shape (1, 1, 250)
            
        Returns:
            dict: Dictionary mapping vulnerability types to detection results
        """
        results = {}
        
        for vul_type, model in self.models.items():
            try:
                # Make prediction
                prediction = model.model.predict(
                    [graph_features, pattern_features[0], pattern_features[1], pattern_features[2]]
                )
                
                # Convert to boolean (vulnerable or not)
                is_vulnerable = bool(prediction > 0.5)
                
                # Get feature importance weights
                graphweight_model = tf.keras.Model(
                    inputs=model.model.input, 
                    outputs=model.model.get_layer('outputgraphweight').output
                )
                graphweight = graphweight_model.predict(
                    [graph_features, pattern_features[0], pattern_features[1], pattern_features[2]]
                )[0][0][0]
                
                pattern1weight_model = tf.keras.Model(
                    inputs=model.model.input, 
                    outputs=model.model.get_layer('outputpattern1weight').output
                )
                pattern1weight = pattern1weight_model.predict(
                    [graph_features, pattern_features[0], pattern_features[1], pattern_features[2]]
                )[0][0][0]
                
                pattern2weight_model = tf.keras.Model(
                    inputs=model.model.input, 
                    outputs=model.model.get_layer('outputpattern2weight').output
                )
                pattern2weight = pattern2weight_model.predict(
                    [graph_features, pattern_features[0], pattern_features[1], pattern_features[2]]
                )[0][0][0]
                
                pattern3weight_model = tf.keras.Model(
                    inputs=model.model.input, 
                    outputs=model.model.get_layer('outputpattern3weight').output
                )
                pattern3weight = pattern3weight_model.predict(
                    [graph_features, pattern_features[0], pattern_features[1], pattern_features[2]]
                )[0][0][0]
                
                # Store results
                results[vul_type] = {
                    'is_vulnerable': is_vulnerable,
                    'confidence': float(prediction[0][0]),
                    'feature_weights': {
                        'graph': float(graphweight),
                        'pattern1': float(pattern1weight),
                        'pattern2': float(pattern2weight),
                        'pattern3': float(pattern3weight)
                    }
                }
            except Exception as e:
                results[vul_type] = {
                    'error': str(e)
                }
        
        return results

# Example usage:
if __name__ == "__main__":
    # Initialize the detector
    detector = VulnerabilityDetector()
    
    # Example feature extraction (this would come from a real smart contract)
    graph_features = np.random.random((1, 1, 250))
    pattern1 = np.random.random((1, 1, 250))
    pattern2 = np.random.random((1, 1, 250))
    pattern3 = np.random.random((1, 1, 250))
    
    # Detect vulnerabilities
    results = detector.detect_vulnerabilities(graph_features, [pattern1, pattern2, pattern3])
    
    # Print results
    for vul_type, result in results.items():
        if 'error' in result:
            print(f"{vul_type}: Error - {result['error']}")
        else:
            print(f"{vul_type}: {'Vulnerable' if result['is_vulnerable'] else 'Safe'} (Confidence: {result['confidence']:.4f})")
            print(f"  Feature weights:")
            for feature, weight in result['feature_weights'].items():
                print(f"    {feature}: {weight:.4f}") 