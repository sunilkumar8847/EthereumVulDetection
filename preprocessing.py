import numpy as np
import os
import re
import logging
import random
from scipy.signal import resample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pattern_feature(vulnerability_type, train_size, test_size):
    """
    Load pattern features from files and ensure they match the required sizes.
    
    Args:
        vulnerability_type (str): Type of vulnerability ('reentrancy' or 'timestamp')
        train_size (int): Number of training samples needed
        test_size (int): Number of testing samples needed
        
    Returns:
        tuple: (train_pattern_features, test_pattern_features) with shapes (train_size, 250) and (test_size, 250)
    """
    pattern_dir = f'./pattern_feature/feature_FNN/{vulnerability_type}'
    pattern_features = []
    
    try:
        # Load all pattern files
        for filename in os.listdir(pattern_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(pattern_dir, filename)
                with open(file_path, 'r') as f:
                    # Read all lines and convert to float arrays
                    for line in f:
                        try:
                            # Convert line to float array
                            feature = np.array([float(x) for x in line.strip().split()])
                            
                            # Enhanced feature processing for reentrancy
                            if vulnerability_type == 'reentrancy':
                                # Ensure exact length of 250
                                if len(feature) > 250:
                                    # Use average pooling for reduction
                                    n_segments = len(feature) // 250
                                    feature = np.mean(feature[:n_segments*250].reshape(-1, n_segments), axis=1)
                                elif len(feature) < 250:
                                    # Use interpolation for upsampling
                                    indices = np.linspace(0, len(feature)-1, 250)
                                    feature = np.interp(indices, np.arange(len(feature)), feature)
                            else:
                                # For other vulnerabilities, use original padding/truncation
                                if len(feature) > 250:
                                    feature = feature[:250]
                                elif len(feature) < 250:
                                    feature = np.pad(feature, (0, 250 - len(feature)))
                            
                            # Normalize feature vector
                            feature = (feature - np.mean(feature)) / (np.std(feature) + 1e-8)
                            
                            pattern_features.append(feature)
                        except Exception as e:
                            print(f"Error processing line in {filename}: {str(e)}")
                            continue
        
        if not pattern_features:
            print(f"No valid pattern features loaded from {pattern_dir}")
            return None, None
            
        # Convert to numpy array
        pattern_features = np.array(pattern_features)
        print(f"Loaded {len(pattern_features)} pattern features with shape {pattern_features.shape}")
        
        # Calculate total size needed
        total_size = train_size + test_size
        
        # Enhanced data handling for reentrancy with sophisticated augmentation
        if vulnerability_type == 'reentrancy':
            if len(pattern_features) < total_size:
                # Use advanced data augmentation techniques
                n_orig = len(pattern_features)
                n_needed = total_size - n_orig
                
                logger.info(f"Performing advanced augmentation for reentrancy patterns - creating {n_needed} new samples")
                augmented = []
                
                # Calculate feature complexity metrics to identify complex patterns
                feature_complexity = np.zeros(n_orig)
                for i in range(n_orig):
                    # Calculate complexity metrics (variance, peaks, etc.)
                    feature_complexity[i] = np.var(pattern_features[i]) * 10 + np.sum(np.abs(np.diff(pattern_features[i])))
                
                # Normalize complexity scores for weighted sampling
                complexity_weights = feature_complexity / np.sum(feature_complexity)
                
                # Create different types of augmented samples
                for _ in range(n_needed):
                    augmentation_type = np.random.choice(['noise', 'synthetic', 'hybrid'], 
                                                        p=[0.4, 0.3, 0.3])
                    
                    if augmentation_type == 'noise':
                        # Add controlled noise - more noise for less complex patterns
                        idx = np.random.choice(n_orig, p=complexity_weights)
                        base_feature = pattern_features[idx]
                        noise_scale = 0.05 + 0.1 * (1 - feature_complexity[idx] / np.max(feature_complexity))
                        noise = np.random.normal(0, noise_scale, base_feature.shape)
                        augmented.append(base_feature + noise)
                        
                    elif augmentation_type == 'synthetic':
                        # Create synthetic examples focused on critical reentrancy patterns
                        # Select 2-3 high complexity samples and blend them
                        n_samples = np.random.randint(2, 4)
                        # Higher probability for complex patterns
                        idxs = np.random.choice(n_orig, size=n_samples, p=complexity_weights)
                        weights = np.random.dirichlet(np.ones(n_samples))
                        
                        synthetic = np.zeros(250)
                        for i, idx in enumerate(idxs):
                            synthetic += weights[i] * pattern_features[idx]
                            
                        # Enhance specific ranges that are important for reentrancy
                        # (based on domain knowledge that certain feature ranges are more important)
                        important_ranges = [(50, 80), (120, 150), (200, 230)]
                        for start, end in important_ranges:
                            # Amplify important sections
                            if np.random.random() < 0.7:  # 70% chance to enhance
                                amplification = 1.2 + 0.3 * np.random.random()
                                synthetic[start:end] *= amplification
                                
                        augmented.append(synthetic)
                        
                    else:  # hybrid approach
                        # Combine techniques - warp existing pattern
                        idx = np.random.choice(n_orig, p=complexity_weights)
                        base_feature = pattern_features[idx].copy()
                        
                        # Apply random time warping to sections
                        n_sections = np.random.randint(2, 5)
                        section_size = 250 // n_sections
                        for i in range(n_sections):
                            start = i * section_size
                            end = (i + 1) * section_size if i < n_sections - 1 else 250
                            
                            # Either compress or stretch this section
                            warp_factor = 0.8 + 0.4 * np.random.random()  # 0.8-1.2
                            new_size = int((end - start) * warp_factor)
                            section = base_feature[start:end]
                            
                            # Resample this section
                            warped_section = resample(section, new_size)
                            
                            # Insert back, padding or truncating as needed
                            if new_size <= (end - start):
                                # Pad if smaller
                                padding = np.zeros((end - start) - new_size)
                                base_feature[start:end] = np.concatenate([warped_section, padding])
                            else:
                                # Truncate if larger
                                base_feature[start:end] = warped_section[:(end-start)]
                                
                        # Add minor noise
                        noise = np.random.normal(0, 0.03, base_feature.shape)
                        augmented.append(base_feature + noise)
                
                # Combine original and augmented samples
                pattern_features = np.vstack([pattern_features, augmented])
                logger.info(f"Augmentation complete. New dataset size: {len(pattern_features)}")
                
            elif len(pattern_features) > total_size:
                # Strategic sampling to maintain class balance and feature diversity
                # Calculate feature diversity to sample diverse patterns
                from sklearn.cluster import KMeans
                
                # Use K-means to identify clusters of features
                n_clusters = min(10, len(pattern_features) // 50)  # Adjust number of clusters
                kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pattern_features)
                clusters = kmeans.labels_
                
                # Sample proportionally from each cluster
                indices = []
                for i in range(n_clusters):
                    cluster_indices = np.where(clusters == i)[0]
                    cluster_size = len(cluster_indices)
                    # Calculate how many samples to take from this cluster
                    n_samples = max(1, int(total_size * (cluster_size / len(pattern_features))))
                    if len(indices) + n_samples > total_size:
                        n_samples = total_size - len(indices)
                    
                    # Sample from this cluster
                    if n_samples > 0:
                        sampled = np.random.choice(cluster_indices, size=n_samples, replace=False)
                        indices.extend(sampled)
                    
                    if len(indices) >= total_size:
                        break
                
                # If we still need more samples, add random ones
                if len(indices) < total_size:
                    remaining = np.setdiff1d(np.arange(len(pattern_features)), indices)
                    additional = np.random.choice(remaining, size=total_size-len(indices), replace=False)
                    indices.extend(additional)
                
                pattern_features = pattern_features[indices[:total_size]]
        else:
            # For other vulnerabilities, use original handling
            if len(pattern_features) < total_size:
                repeat_times = int(np.ceil(total_size / len(pattern_features)))
                pattern_features = np.tile(pattern_features, (repeat_times, 1))
                pattern_features = pattern_features[:total_size]
            elif len(pattern_features) > total_size:
                indices = np.random.permutation(len(pattern_features))[:total_size]
                pattern_features = pattern_features[indices]
        
        # Split into train and test
        train_pattern_features = pattern_features[:train_size]
        test_pattern_features = pattern_features[train_size:train_size + test_size]
        
        # Final normalization
        train_mean = np.mean(train_pattern_features, axis=0)
        train_std = np.std(train_pattern_features, axis=0) + 1e-8
        
        train_pattern_features = (train_pattern_features - train_mean) / train_std
        test_pattern_features = (test_pattern_features - train_mean) / train_std
        
        print(f"Final pattern feature shapes - Train: {train_pattern_features.shape}, Test: {test_pattern_features.shape}")
        return train_pattern_features, test_pattern_features
        
    except Exception as e:
        print(f"Error loading pattern features: {str(e)}")
        return None, None


def load_single_pattern_feature(file_path):
    """Load pattern features from a single file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            features = []
            for line in lines:
                line_features = [float(x) for x in line.strip().split()]
                features.append(line_features)
            
            features = np.array(features)
            
            # Ensure correct shape (250 dimensions)
            if features.shape[1] != 250:
                logger.warning(f"Pattern in {file_path} has wrong length: {features.shape[1]}, fixing")
                if features.shape[1] < 250:
                    padding = np.zeros((features.shape[0], 250 - features.shape[1]))
                    features = np.hstack([features, padding])
                else:
                    features = features[:, :250]
            
            return features
    except Exception as e:
        logger.error(f"Error loading pattern feature for {file_path}: {str(e)}")
        return None


def get_graph_feature(vulnerability_type):
    """Load and preprocess graph features."""
    try:
        train_path = f'./graph_feature/{vulnerability_type}/{vulnerability_type}_final_train.txt'
        test_path = f'./graph_feature/{vulnerability_type}/{vulnerability_type}_final_valid.txt'
        
        # Load training data
        if os.path.exists(train_path):
            try:
                # First try loading with whitespace delimiter
                train_data = np.loadtxt(train_path, delimiter=' ')
            except:
                # If that fails, try comma delimiter
                train_data = np.loadtxt(train_path, delimiter=',')
            
            # For timestamp vulnerability, handle larger feature size
            if vulnerability_type == 'timestamp':
                # Each line is a feature vector of size 250
                n_samples = train_data.shape[0]
                # Take only the first 1000 samples if there are more
                if n_samples > 1000:
                    train_data = train_data[:1000]
                # Add the extra dimension for GNN
                train_data = train_data.reshape(-1, 1, 250)
            else:
                # For reentrancy, keep original reshape
                train_data = train_data.reshape(-1, 1, 250)
        else:
            logging.error(f"Training file not found: {train_path}")
            train_data = np.zeros((1, 1, 250))  # Dummy data
            
        # Load testing data
        if os.path.exists(test_path):
            try:
                # First try loading with whitespace delimiter
                test_data = np.loadtxt(test_path, delimiter=' ')
            except:
                # If that fails, try comma delimiter
                test_data = np.loadtxt(test_path, delimiter=',')
            
            # For timestamp vulnerability, handle larger feature size
            if vulnerability_type == 'timestamp':
                # Each line is a feature vector of size 250
                n_samples = test_data.shape[0]
                # Take only the first 250 samples if there are more
                if n_samples > 250:
                    test_data = test_data[:250]
                # Add the extra dimension for GNN
                test_data = test_data.reshape(-1, 1, 250)
            else:
                # For reentrancy, keep original reshape
                test_data = test_data.reshape(-1, 1, 250)
        else:
            logging.error(f"Testing file not found: {test_path}")
            test_data = np.zeros((1, 1, 250))  # Dummy data
            
        logging.info(f"Graph features loaded - Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        return train_data, test_data
        
    except Exception as e:
        logging.error(f"Error loading graph features: {str(e)}")
        raise


def ensure_result_dirs():
    """Ensure that result directories exist for all vulnerability types"""
    for vul_type in ['reentrancy', 'timestamp']:
        os.makedirs(f"results/{vul_type}", exist_ok=True)


if __name__ == "__main__":
    # Test both vulnerability types
    for vul_type in ['reentrancy', 'timestamp']:
        print(f"Testing {vul_type} vulnerability data loading...")
        try:
            pattern_train, pattern_test = get_pattern_feature(vul_type, 100, 100)
            graph_train, graph_test = get_graph_feature(vul_type)
            print(f"Successfully loaded {vul_type} data")
            print(f"Train data shape: {graph_train.shape}")
            print(f"Test data shape: {graph_test.shape}")
        except Exception as e:
            print(f"Error loading {vul_type} data: {str(e)}")

