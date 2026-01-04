"""
Data loading utilities for leukemia blood cell classification.
"""
import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(
    data_dir: str,
    img_size: tuple = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    augment: bool = False,
    seed: int = 42
):
    """
    Load and preprocess blood cell images from directory structure.
    
    Args:
        data_dir: Path to data directory containing training_data/fold_*/
        img_size: Target image size (height, width)
        batch_size: Batch size for generators
        validation_split: Fraction of data to use for validation
        augment: Whether to apply data augmentation to training data
        seed: Random seed for reproducibility
    
    Returns:
        train_generator, validation_generator, test_generator
    """
    data_path = Path(data_dir) / "training_data"
    
    # Combine all folds into single directory structure
    # Expected: fold_0/all, fold_0/hem, fold_1/all, fold_1/hem, etc.
    
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # For simplicity, use fold_0 and fold_1 for train/val, fold_2 for test
    train_val_dir = data_path / "fold_0"
    test_dir = data_path / "fold_2"
    
    train_generator = train_datagen.flow_from_directory(
        train_val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        seed=seed
    )
    
    val_generator = train_datagen.flow_from_directory(
        train_val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        seed=seed
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def get_sample_images(data_dir: str, n_samples: int = 5):
    """
    Get sample images from each class for visualization.
    
    Args:
        data_dir: Path to data directory
        n_samples: Number of samples per class
    
    Returns:
        dict with 'all' and 'hem' keys, each containing list of image paths
    """
    import random
    
    data_path = Path(data_dir) / "training_data" / "fold_0"
    samples = {'all': [], 'hem': []}
    
    for label in ['all', 'hem']:
        label_path = data_path / label
        if label_path.exists():
            images = list(label_path.glob('*.bmp')) + list(label_path.glob('*.jpg'))
            samples[label] = random.sample(images, min(n_samples, len(images)))
    
    return samples
