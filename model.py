import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout, Concatenate, Reshape, GlobalMaxPooling1D, BatchNormalization, Multiply
from tensorflow.keras.models import Model
import numpy as np
import logging
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall, AUC

class GNNLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', kernel_regularizer=None, **kwargs):
        super(GNNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = kernel_regularizer or tf.keras.regularizers.l2(0.01)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            regularizer=self.kernel_regularizer,
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias'
        )
        super(GNNLayer, self).build(input_shape)
        
    def call(self, inputs):
        # Apply graph convolution
        output = tf.matmul(inputs, self.kernel)
        output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

def process_graph_feature(train_data, test_data=None):
    """Process graph features for training and testing."""
    # Convert to numpy arrays if they aren't already
    train_data = np.array(train_data)
    
    # Reshape to ensure we have the correct dimensions (batch_size, 1, feature_dim)
    if len(train_data.shape) == 2:
        train_data = np.expand_dims(train_data, axis=1)
    
    logging.info(f"Graph feature train shape: {train_data.shape}")
    
    if test_data is not None:
        test_data = np.array(test_data)
        if len(test_data.shape) == 2:
            test_data = np.expand_dims(test_data, axis=1)
        logging.info(f"Graph feature test shape: {test_data.shape}")
        return train_data, test_data
    
    return train_data

def process_pattern_feature(train_data, test_data=None):
    """Process pattern features for training and testing."""
    # Convert to numpy arrays if they aren't already
    train_data = np.array(train_data)
    
    # Ensure we have the correct shape (batch_size, feature_dim)
    if len(train_data.shape) > 2:
        train_data = train_data.reshape(train_data.shape[0], -1)
    
    logging.info(f"Pattern feature train shape: {train_data.shape}")
    
    if test_data is not None:
        test_data = np.array(test_data)
        if len(test_data.shape) > 2:
            test_data = test_data.reshape(test_data.shape[0], -1)
        logging.info(f"Pattern feature test shape: {test_data.shape}")
        return train_data, test_data
    
    return train_data

def build_model():
    """Build the vulnerability detection model."""
    # Graph feature input (batch_size, 1, 250)
    graph_input = Input(shape=(1, 250), name='graph_input')
    
    # Pattern feature input (batch_size, 250)
    pattern_input = Input(shape=(250,), name='pattern_input')
    
    # Graph Neural Network branch
    x = GNNLayer(128, activation='relu')(graph_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = GNNLayer(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Add a third GNN layer for deeper pattern matching - especially beneficial for reentrancy
    x = GNNLayer(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)  # Slightly lower dropout to avoid over-regularization
    
    # Add attention mechanism to focus on important parts of the graph
    attention_weights = Dense(32, activation='softmax', name='attention_weights')(x)
    x = Multiply()([x, attention_weights])
    
    x = GlobalMaxPooling1D()(x)
    
    # Pattern feature branch
    y = Dense(128, activation='relu')(pattern_input)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    
    y = Dense(64, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    
    # Concatenate branches
    combined = Concatenate()([x, y])
    
    # Dense layers
    z = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.3)(z)
    
    z = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(z)
    z = BatchNormalization()(z)
    z = Dropout(0.3)(z)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(z)
    
    # Create model
    model = Model(inputs=[graph_input, pattern_input], outputs=output)
    
    # Create learning rate scheduler with decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0005,  # Lower initial learning rate
        decay_steps=1000,
        decay_rate=0.95,
        staircase=True
    )
    
    # Compile model with additional metrics
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    
    return model 