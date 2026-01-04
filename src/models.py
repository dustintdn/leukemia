"""
Model architectures for leukemia blood cell classification.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.applications import VGG16, EfficientNetB3
from tensorflow.keras.optimizers import Adam


def build_cnn(
    filters: int = 32,
    input_shape: tuple = (224, 224, 3),
    learning_rate: float = 0.001
):
    """
    Build a custom CNN for binary classification.
    
    Alternates between kernel sizes 3 and 1 across 4 conv layers,
    as used in the original experiments.
    
    Args:
        filters: Number of filters in conv layers
        input_shape: Input image shape
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Layer 1
        Conv2D(filters, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Layer 2
        Conv2D(filters, (1, 1), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Layer 3
        Conv2D(filters, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Layer 4
        Conv2D(filters, (1, 1), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Classifier
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_cnn_with_dropout(
    input_shape: tuple = (224, 224, 3),
    learning_rate: float = 0.0001,
    dropout_rate: float = 0.2
):
    """
    Build a CNN with dropout regularization.
    
    Uses decreasing filter sizes (32 -> 16 -> 8) with dropout
    after the second conv block.
    
    Args:
        input_shape: Input image shape
        learning_rate: Learning rate for Adam optimizer
        dropout_rate: Dropout probability
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Block 1 - 32 filters
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Block 2 - 16 filters with dropout
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        
        # Block 3 - 8 filters
        Conv2D(8, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Classifier
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_vgg16(
    input_shape: tuple = (224, 224, 3),
    learning_rate: float = 0.001,
    freeze_base: bool = True
):
    """
    Build a VGG16-based transfer learning model.
    
    Note: In original experiments, this converged at 50% accuracy
    (random guessing). Consider fine-tuning or using a different
    pretrained model.
    
    Args:
        input_shape: Input image shape
        learning_rate: Learning rate for Adam optimizer
        freeze_base: Whether to freeze VGG16 base weights
    
    Returns:
        Compiled Keras model
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D
    
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    if freeze_base:
        base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_efficientnet(
    input_shape: tuple = (224, 224, 3),
    learning_rate: float = 0.001,
    freeze_base: bool = True
):
    """
    Build an EfficientNetB3-based transfer learning model.
    
    Best performing model in experiments (98% accuracy), but
    showed high loss indicating potential overfitting.
    
    Args:
        input_shape: Input image shape
        learning_rate: Learning rate for Adam optimizer
        freeze_base: Whether to freeze EfficientNet base weights
    
    Returns:
        Compiled Keras model
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D
    
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    if freeze_base:
        base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Model registry for easy access
MODELS = {
    'cnn_32': lambda: build_cnn(filters=32),
    'cnn_16': lambda: build_cnn(filters=16),
    'cnn_dropout': build_cnn_with_dropout,
    'vgg16': build_vgg16,
    'efficientnet': build_efficientnet,
}


def get_model(name: str, **kwargs):
    """
    Get a model by name.
    
    Args:
        name: Model name ('cnn_32', 'cnn_16', 'cnn_dropout', 'vgg16', 'efficientnet')
        **kwargs: Additional arguments passed to model builder
    
    Returns:
        Compiled Keras model
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    
    return MODELS[name](**kwargs)
