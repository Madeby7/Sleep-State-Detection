import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
import numpy as np
from tensorflow.keras.layers import Input, Lambda

from tensorflow.keras.optimizers import Adam
#import tensorflow_addons as tfa
from tensorflow.keras.utils import plot_model
from tensorflow import keras

def build_stacked_model(input_shape, num_classes, model_config):
    """
    Builds a classification model using early fusion with BiLSTM for sensor data.

    Args:
    input_shape (tuple): Shape of the input data after stacking (e.g., (1500, 3, 3)).
    model_config (dict): Dictionary containing all the model configurations and hyperparameters.

    Returns:
    tf.keras.Model: The compiled classification model.
    """
    inputs = Input(shape=input_shape)
    
    x = layers.Conv2D(model_config['num_filters_1'], (3, 3), padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(model_config['num_filters_2'], (3, 3), padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)

    x = layers.Conv2D(model_config['num_filters_3'], (3, 3), padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)

    # Reshape for BiLSTM
    # Convert to shape compatible with LSTM input: (batch_size, time_steps, features)
    # _, time_steps, features, channels = x.shape.as_list()
    _, time_steps, features, channels = list(x.shape)
    reshaped = layers.Reshape((time_steps, features * channels))(x)

    # BiLSTM layers for temporal dependency extraction
    x = layers.Bidirectional(layers.LSTM(model_config['lstm_units'], return_sequences=True))(reshaped)
    x = layers.GlobalMaxPooling1D()(x)

    # Fully connected layers
    x = layers.Dense(model_config['dense_units'], activation='relu')(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    # Compile the model
    initial_learning_rate = model_config['lr']
    decay_steps = model_config['lr_decay_steps']

    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=model_config['lr_alpha']
    )
    optimizer = Adam(learning_rate=lr_schedule)

    # Loss function
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(gamma=model_config['loss_gamma'], alpha=model_config['loss_alpha'])

    # Create and compile the model
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=[keras.metrics.AUC(),
                           keras.metrics.SpecificityAtSensitivity(0.99),
                           keras.metrics.SensitivityAtSpecificity(0.99)])

    return model

def attention_block(inputs):
    attention_probs = layers.Dense(inputs.shape[-1], activation='softmax')(inputs)
    attention_mul = layers.multiply([inputs, attention_probs])
    return attention_mul

def transformer_block(inputs, num_heads, ff_dim, dropout_rate):
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    
    # Add & normalize
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)
    
    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation="relu")(attention_output)
    ff_output = layers.Dense(inputs.shape[-1])(ff_output)
    
    # Add & normalize
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    output = layers.LayerNormalization(epsilon=1e-6)(ff_output + attention_output)
    
    return output
def build_sensor_model_2d_TRANS(input_shape, model_config):
    """
    Builds a sub-model for processing sensor data.

    Args:
    input_shape (tuple): Shape of the input data for the sensor.
    model_config (dict): Dictionary containing all the model configurations and hyperparameters.

    Returns:
    tf.keras.Model: The sub-model for the sensor.
    """
    inputs = Input(shape=input_shape)
    
    x = layers.Conv2D(model_config['num_filters_1'], model_config['kernel_size_1'], padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(model_config['num_filters_2'], model_config['kernel_size_2'], padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.MaxPooling2D(model_config['pooling_size'])(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    
    x = layers.Conv2D(model_config['num_filters_3'], model_config['kernel_size_2'], padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D(model_config['pooling_size'])(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    
    batch_size, height, width, channels = x.shape.as_list()
    new_shape = (height, width * channels)
    
    x = layers.Reshape(new_shape)(x)
    
    # Transformer block
    x = transformer_block(x, num_heads=model_config['num_heads'], 
                          ff_dim=model_config['ff_dim'], dropout_rate=model_config['dropout_rate'])
    
    # LSTM block or dense layers
    x = layers.Bidirectional(layers.LSTM(model_config['lstm_units'], return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)
    
    return Model(inputs, x)

def build_sensor_model_2d_2(input_shape, model_config):
    """
    Builds a sub-model for processing sensor data.

    Args:
    input_shape (tuple): Shape of the input data for the sensor.
    model_config (dict): Dictionary containing all the model configurations and hyperparameters.

    Returns:
    tf.keras.Model: The sub-model for the sensor.
    """
    inputs = Input(shape=input_shape)
    
    x = layers.Conv2D(model_config['num_filters_1'], model_config['kernel_size_1'], padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(model_config['num_filters_2'], model_config['kernel_size_2'], padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.MaxPooling2D(model_config['pooling_size'])(x)
    # Removed Dropout here to avoid applying it on a 4D tensor

    x = layers.Conv2D(model_config['num_filters_3'], model_config['kernel_size_2'], padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D(model_config['pooling_size'])(x)
    # Removed Dropout here as well

    # Attention block
    # x = attention_block(x)
    
    # Use Reshape to prepare for LSTM layer
    shape_before_lstm = x.shape[1] * x.shape[2]  # Combine height and width
    x = layers.Reshape((shape_before_lstm, x.shape[-1]))(x)  
    x = layers.Dropout(model_config['dropout_rate'])(x)  # Apply Dropout after Reshape


    x = layers.Bidirectional(layers.LSTM(model_config['lstm_units'], return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)

    return Model(inputs, x)

def build_sensor_model_2d_sm_2(input_shape, model_config):
    """
    Builds a sub-model for processing sensor data with simpler structure.

    Args:
    input_shape (tuple): Shape of the input data for the sensor.
    model_config (dict): Dictionary containing all the model configurations and hyperparameters.

    Returns:
    tf.keras.Model: The sub-model for the sensor.
    """
    inputs = Input(shape=input_shape)
    
    x = layers.Conv2D(model_config['num_filters_1_sm'], model_config['kernel_size_1_sm'], padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.MaxPooling2D(model_config['pooling_size'])(x)
    # Removed Dropout here as well

    # Flatten and Reshape to prepare for Dense and GlobalMaxPooling1D
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Reshape to add a dimension for compatibility with GlobalMaxPooling1D
    x = layers.Reshape((32, 1))(x)  # Reshaping to (batch, 32, 1)
    x = layers.GlobalMaxPooling1D()(x)

    return Model(inputs, x)

def build_sensor_model_2d(input_shape, model_config):
    """
    Builds a sub-model for processing sensor data.

    Args:
    input_shape (tuple): Shape of the input data for the sensor.
    model_config (dict): Dictionary containing all the model configurations and hyperparameters.

    Returns:
    tf.keras.Model: The sub-model for the sensor.
    """
    inputs = Input(shape=input_shape)
    
    x = layers.Conv2D(model_config['num_filters_1'], model_config['kernel_size_1'], padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(model_config['num_filters_2'], model_config['kernel_size_2'], padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.MaxPooling2D(model_config['pooling_size'])(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    
    x = layers.Conv2D(model_config['num_filters_3'], model_config['kernel_size_2'], padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D(model_config['pooling_size'])(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    
    # Attention block
    #x = attention_block(x)
    
    batch_size, height, width, channels = x.shape.as_list()
    new_shape = (height, width * channels)
    
    x = layers.Reshape(new_shape)(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    x = layers.Bidirectional(layers.LSTM(model_config['lstm_units'], return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)

    return Model(inputs, x)

def build_sensor_model_2d_sm(input_shape, model_config):
    """
    Builds a sub-model for processing sensor data.

    Args:
    input_shape (tuple): Shape of the input data for the sensor.
    model_config (dict): Dictionary containing all the model configurations and hyperparameters.

    Returns:
    tf.keras.Model: The sub-model for the sensor.
    """
    inputs = Input(shape=input_shape)
    
    x = layers.Conv2D(model_config['num_filters_1_sm'], model_config['kernel_size_1_sm'], padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(model_config['l2_regularization']),
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.MaxPooling2D(model_config['pooling_size'])(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    
    # Attention block
    #x = attention_block(x)
    
    batch_size, height, width, channels = x.shape.as_list()
    new_shape = (height, width * channels)
    
    x = layers.Reshape(new_shape)(x)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    x = layers.Dense(32, activation='relu')(x)
    #x = layers.Bidirectional(layers.LSTM(model_config['lstm_units_sm'], return_sequences=True))(x)
    x = layers.GlobalMaxPooling1D()(x)

    return Model(inputs, x)

def build_combined_model(input_shapes, num_classes, model_config): # late fusion model architecture
    """
    Builds a classification model that combines data from three sensors.

    Args:
    input_shapes (list of tuples): Shapes of the input data for each sensor.
    num_classes (int): Number of output classes.
    model_config (dict): Dictionary containing all the model configurations and hyperparameters.

    Returns:
    tf.keras.Model: The compiled classification model.
    """
    # Create sub-models for each sensor
#     sensor1_model = build_sensor_model_2d(input_shapes[0], model_config)
#     sensor2_model = build_sensor_model_2d(input_shapes[1], model_config)
#     sensor3_model = build_sensor_model_2d(input_shapes[2], model_config)
#     sensor4_model = build_sensor_model_2d_sm(input_shapes[3], model_config)

    sensor1_model = build_sensor_model_2d_2(input_shapes[0], model_config)
    sensor2_model = build_sensor_model_2d_2(input_shapes[1], model_config)
    sensor3_model = build_sensor_model_2d_2(input_shapes[2], model_config)
    sensor4_model = build_sensor_model_2d_sm_2(input_shapes[3], model_config)
    
    #sensor1_model = build_sensor_model_2d_TRANS(input_shapes[0], model_config)
    #sensor2_model = build_sensor_model_2d_TRANS(input_shapes[1], model_config)
    #sensor3_model = build_sensor_model_2d_TRANS(input_shapes[2], model_config)
    #sensor4_model = build_sensor_model_2d_sm(input_shapes[3], model_config)
    
    
    # Define inputs for each sensor
    sensor1_input = Input(shape=input_shapes[0])
    sensor2_input = Input(shape=input_shapes[1])
    sensor3_input = Input(shape=input_shapes[2])
    sensor4_input = Input(shape=input_shapes[3])

    # Get the outputs from each sub-model
    sensor1_output = sensor1_model(sensor1_input)
    sensor2_output = sensor2_model(sensor2_input)
    sensor3_output = sensor3_model(sensor3_input)
    sensor4_output = sensor4_model(sensor4_input)

    # Concatenate the outputs
    concatenated = layers.concatenate([sensor1_output, sensor2_output, sensor3_output, sensor4_output])

    # Add Dense and Dropout layers
    x = layers.Dense(model_config['dense_units'], activation='relu')(concatenated)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    #output = layers.Dense(1)(x)

    initial_learning_rate = model_config['lr']
    decay_steps = model_config['lr_decay_steps']  

    # Create the learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=model_config['lr_alpha']
        
    )

    optimizer = Adam(learning_rate=lr_schedule)
    
    # Create and compile the model
    model = Model(inputs=[sensor1_input, sensor2_input, sensor3_input, sensor4_input], outputs=output)
    
    #loss_fn = tf.keras.losses.BinaryCrossentropy()
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(gamma=model_config['loss_gamma'], alpha=model_config['loss_alpha'])
    
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=[keras.metrics.AUC(),
                             keras.metrics.SpecificityAtSensitivity(0.99),
                             keras.metrics.SensitivityAtSpecificity(0.99)]
                 )
    
    

    return model



    
def build_combined_model_removed_1(input_shapes, num_classes, model_config):
    """
    Builds a classification model that combines data from three sensors.

    Args:
    input_shapes (list of tuples): Shapes of the input data for each sensor.
    num_classes (int): Number of output classes.
    model_config (dict): Dictionary containing all the model configurations and hyperparameters.

    Returns:
    tf.keras.Model: The compiled classification model.
    """
    # Create sub-models for each sensor
#     sensor1_model = build_sensor_model_2d(input_shapes[0], model_config)
#     sensor2_model = build_sensor_model_2d(input_shapes[1], model_config)
#     sensor3_model = build_sensor_model_2d(input_shapes[2], model_config)
#     sensor4_model = build_sensor_model_2d_sm(input_shapes[3], model_config)

    sensor1_model = build_sensor_model_2d_2(input_shapes[0], model_config)
    sensor2_model = build_sensor_model_2d_2(input_shapes[1], model_config)
    sensor3_model = build_sensor_model_2d_2(input_shapes[2], model_config)
    sensor4_model = build_sensor_model_2d_sm_2(input_shapes[3], model_config)
    
    #sensor1_model = build_sensor_model_2d_TRANS(input_shapes[0], model_config)
    #sensor2_model = build_sensor_model_2d_TRANS(input_shapes[1], model_config)
    #sensor3_model = build_sensor_model_2d_TRANS(input_shapes[2], model_config)
    #sensor4_model = build_sensor_model_2d_sm(input_shapes[3], model_config)
    
    
    # Define inputs for each sensor
    sensor1_input = Input(shape=input_shapes[0])
    sensor2_input = Input(shape=input_shapes[1])
    sensor3_input = Input(shape=input_shapes[2])
    sensor4_input = Input(shape=input_shapes[3])

    # Get the outputs from each sub-model
    sensor1_output = sensor1_model(sensor1_input)
    # sensor2_output = sensor2_model(sensor2_input)
    # sensor3_output = sensor3_model(sensor3_input)
    # sensor4_output = sensor4_model(sensor4_input)

    # Concatenate the outputs
    # concatenated = layers.concatenate([sensor1_output, sensor2_output, sensor3_output, sensor4_output])
    concatenated = sensor1_output
    
    # Add Dense and Dropout layers
    x = layers.Dense(model_config['dense_units'], activation='relu')(concatenated)
    x = layers.Dropout(model_config['dropout_rate'])(x)
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    #output = layers.Dense(1)(x)

    initial_learning_rate = model_config['lr']
    decay_steps = model_config['lr_decay_steps']  

    # Create the learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=model_config['lr_alpha']
        
    )

    optimizer = Adam(learning_rate=lr_schedule)
    
    # Create and compile the model
    model = Model(inputs=[sensor1_input, sensor2_input, sensor3_input, sensor4_input], outputs=output)
    
    #loss_fn = tf.keras.losses.BinaryCrossentropy()
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(gamma=model_config['loss_gamma'], alpha=model_config['loss_alpha'])
    
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=[keras.metrics.AUC(),
                             keras.metrics.SpecificityAtSensitivity(0.99),
                             keras.metrics.SensitivityAtSpecificity(0.99)]
                 )
    
    

    return model


    
def feature_model_1D(input_dim,model_config):
    inputs = Input(shape=(input_dim,), name="input_features")
    
    # First Dense Layer
    x = tf.keras.layers.Dense(128, activation='relu', name="dense_1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="batch_norm_1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    
    # Second Dense Layer
    x = tf.keras.layers.Dense(64, activation='relu', name="dense_2")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm_2")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)
    
    # Third Dense Layer
    x = tf.keras.layers.Dense(32, activation='relu', name="dense_3")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm_3")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_3")(x)
    
    # Output Layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(x)
    
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="sleep_awake_classifier")
    
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(gamma=3, alpha=0.9)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=model_config['lr']),
        loss=loss_fn,
        metrics=[keras.metrics.AUC(),
                 keras.metrics.SpecificityAtSensitivity(0.99),
                 keras.metrics.SensitivityAtSpecificity(0.99)]
    )
    
    return model

def wrap_model_for_single_input(original_model, input_shapes):
    """
    Wraps a multi-input model to accept a single concatenated input tensor.

    Args:
    original_model (tf.keras.Model): The original multi-input model.
    input_shapes (list of tuples): Shapes of the inputs for the original model.

    Returns:
    tf.keras.Model: The wrapped model with a single input tensor.
    """
    # Calculate the total number of features in the concatenated input
    total_features = sum(np.prod(shape) for shape in input_shapes)

    # Define a new single input tensor
    single_input = Input(shape=(total_features,))

    # Split the single input tensor into the original input tensors
    split_tensors = []
    start_idx = 0
    for shape in input_shapes:
        num_features = np.prod(shape)
        end_idx = start_idx + num_features
        # Extract the slice corresponding to this input and reshape
        slice_tensor = single_input[..., start_idx:end_idx]
        reshaped_tensor = tf.reshape(slice_tensor, [-1, *shape])  # Ensure correct reshaping
        split_tensors.append(reshaped_tensor)
        start_idx = end_idx

    # Pass the split inputs to the original model
    output = original_model(split_tensors)

    # Create and return the wrapped model
    wrapped_model = Model(inputs=single_input, outputs=output)
    return wrapped_model

if __name__ == '__main__':
    
#     input_shapes = (1500,3,3)

#     num_classes = 2  # Adjust as per your number of classes

#     model_config = {
#             'lr': 0.001,
#             'num_filters_1': 8,
#             'num_filters_2': 16,
#             'num_filters_3':32,
#             'num_filters_4':64,
#             'num_filters_1_sm': 8,
#             'num_filters_2_sm': 16,
#             'kernel_size_1': (3, 1),
#             'kernel_size_2': (3, 1),
#             'kernel_size_1_sm': (2, 1),
#             'lstm_units': 32,
#             'dropout_rate': 0.2,
#             'dense_units': 64,
#             'l2_regularization': 0.001,
#             'pooling_size': (2, 1),
#             'num_heads': 8,    # Number of heads in Transformer block
#             'ff_dim': 128,     # Feed-forward layer size in Transformer block
#             'loss_gamma': 4,
#             'loss_alpha': 0.2,
#             'lr_alpha': 0.5,
#              'lr_decay_steps': 12100
#         }
    
#     model_config = {
#             'lr': 0.001,
#             'num_filters_1': 16,
#             'num_filters_2': 32,
#             'num_filters_3':64,
#             'dropout_rate': 0.3,
#             'dense_units': 128,
#             'l2_regularization': 0.0001,
#             'loss_gamma': 4,
#             'loss_alpha': 0.2,
#             'lr_alpha': 0.1,
#              'lr_decay_steps': 12100
#         }
    
#     model = build_stacked_model(input_shapes, num_classes, model_config)
#     model.summary()
#     random_data = np.random.rand(1, 1500,3,3).astype(np.float32)
#     output = model.predict(random_data)
    
#     # Define input shape and number of classes
    ppg_input_shape = (1500, 3, 1)  # Adjust as per your data
    gyro_input_shape = (1500, 3, 1)  # Adjust as per your data
    acc_input_shape = (1500, 3, 1)  # Adjust as per your data
    temp_input_shape = (1500, 3, 1)

    input_shapes = [ppg_input_shape, gyro_input_shape, acc_input_shape, temp_input_shape]

    num_classes = 2  # Adjust as per your number of classes

    model_config = {
            'lr': 0.001,
            'num_filters_1': 8,
            'num_filters_2': 16,
            'num_filters_3':32,
            'num_filters_4':64,
            'num_filters_1_sm': 8,
            'num_filters_2_sm': 16,
            'kernel_size_1': (3, 1),
            'kernel_size_2': (3, 1),
            'kernel_size_1_sm': (2, 1),
            'lstm_units': 32,
            'dropout_rate': 0.2,
            'dense_units': 64,
            'l2_regularization': 0.001,
            'pooling_size': (2, 1),
            'num_heads': 8,    # Number of heads in Transformer block
            'ff_dim': 128,     # Feed-forward layer size in Transformer block
             'loss_gamma': 4,
            'loss_alpha': 0.2,
            'lr_alpha': 0.5,
             'lr_decay_steps': 12100
        }
    
    # Build and summarize the model
    model = build_combined_model(input_shapes, num_classes, model_config)

    # Wrap the model for a single concatenated input
    wrapped_model = wrap_model_for_single_input(
        model,
        input_shapes=input_shapes
    )

    # Save the wrapped model
    wrapped_model.save('wrapped_model')
    
    converter = tf.lite.TFLiteConverter.from_saved_model('wrapped_model')
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    
    try:
        tflite_model = converter.convert()
        # Save the TensorFlow Lite model
        with open('wrapped_model.tflite', 'wb') as f:
            f.write(tflite_model)
        print("TensorFlow Lite model conversion successful.")
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
    
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True,expand_nested=True)
    model.summary()

#     # Create synthetic test input data
#     ppg_data = np.random.rand(1, *ppg_input_shape).astype(np.float32)  # Batch size of 1
#     gyro_data = np.random.rand(1, *gyro_input_shape).astype(np.float32)
#     acc_data = np.random.rand(1, *acc_input_shape).astype(np.float32)
#     temp_data = np.random.rand(1, *temp_input_shape).astype(np.float32)

#     # Print shapes of the synthetic data
#     print("PPG data shape:", ppg_data.shape)
#     print("Gyro data shape:", gyro_data.shape)
#     print("Acc data shape:", acc_data.shape)
#     print("Temp data shape:", temp_data.shape)

#     # Pass the synthetic data through the model to get the output
#     output = model.predict([ppg_data, gyro_data, acc_data, temp_data])
#     #output = model.predict([ppg_data, gyro_data, acc_data])

#     # Print the model output
#     print("Model output:", output)
    
#     input_dim = 345  # Replace with the number of features in your dataset
#     model = feature_model_1D(input_dim, model_config)
#     model.summary()
    
    
    