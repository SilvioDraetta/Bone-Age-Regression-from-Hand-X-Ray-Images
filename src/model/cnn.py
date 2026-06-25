import tensorflow as tf

def CNN():
    """
    Creates and compiles a convolutional neural network for image regression.
    The network takes grayscale images as input and extracts spatial features
    through convolutional and pooling layers. The extracted features are then
    mapped to a continuous output value through fully connected layers, making
    the model suitable for regression tasks (e.g., bone age prediction).
    
    Architecture:
        - Input layer: 224x224 grayscale image
        - Three convolutional blocks with ReLU activation
        - MaxPooling layers for spatial downsampling
        - Global Average Pooling to reduce feature dimensions
        - Fully connected layers with dropout regularization
        - Single neuron output for regression
    ---------
    Returns:
        tf.keras.Model:
            A compiled Keras Sequential model ready for training.
    ---------
    Notes:
        The model is compiled using the Adam optimizer with a learning rate
        of 1e-4 and Mean Squared Error (MSE) as loss function. Mean Absolute
        Error (MAE) is used as an additional evaluation metric.
    """
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 1)),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)  # regression output (change as needed)
    ])  
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='mse',
              metrics=['mae'])

    model.summary()
    return model

#DA IMPLEMENTARE
def CNN_MI():
    return 