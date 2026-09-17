"""
Data loading, batching, and augmentation pipelines for TensorFlow/Keras.
Provides both high-performance tf.data pipelines and ImageDataGenerators compatible
across ResNet50, Custom CNN, VGG16, and EfficientNetB3.
"""

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_model_preprocessing_fn(model_name: str) -> Optional[Callable]:
    """
    Returns the appropriate model-specific normalization function.

    Args:
        model_name (str): One of 'ResNet50', 'VGG16', 'EfficientNetB3', 'Custom_CNN'.

    Returns:
        Optional[Callable]: Keras preprocess_input function or None for standard rescale.
    """
    name = model_name.lower().replace("-", "").replace("_", "")
    if "resnet" in name:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    elif "vgg" in name:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input
    elif "efficientnet" in name:
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input
    else:
        # Standard rescale [0, 1] for custom CNN architectures
        return None


def create_image_data_generators(
    dataset_dir: str,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    model_name: str = "ResNet50",
    augmentation_config: Optional[Dict[str, Any]] = None
) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    """
    Constructs train, validation, and test data generators using Keras ImageDataGenerator.

    Guarantees strict isolation between training augmentations and validation/test evaluations.

    Args:
        dataset_dir (str): Root directory containing 'train', 'val', 'test' folders.
        target_size (Tuple[int, int]): Image resolution (width, height).
        batch_size (int): Batch size.
        model_name (str): Target model architecture for normalization.
        augmentation_config (Optional[Dict[str, Any]]): Augmentation parameters.

    Returns:
        Tuple: (train_generator, val_generator, test_generator)
    """
    preprocess_fn = get_model_preprocessing_fn(model_name)
    rescale_val = 1.0 / 255.0 if preprocess_fn is None else None

    # 1. Training Generator with Data Augmentation
    aug_params = {
        "rotation_range": 20,
        "width_shift_range": 0.10,
        "height_shift_range": 0.10,
        "shear_range": 0.10,
        "zoom_range": 0.15,
        "horizontal_flip": True,
        "vertical_flip": False,
        "fill_mode": "nearest",
        "rescale": rescale_val,
        "preprocessing_function": preprocess_fn
    }
    if augmentation_config:
        aug_params.update(augmentation_config)

    train_datagen = ImageDataGenerator(**aug_params)

    # 2. Validation & Test Generator (NO Augmentation, only scaling)
    val_test_datagen = ImageDataGenerator(
        rescale=rescale_val,
        preprocessing_function=preprocess_fn
    )

    train_path = os.path.join(dataset_dir, "train")
    val_path = os.path.join(dataset_dir, "val")
    test_path = os.path.join(dataset_dir, "test")

    train_gen = train_datagen.flow_from_directory(
        directory=train_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    val_gen = val_test_datagen.flow_from_directory(
        directory=val_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        directory=test_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    print(f"\n[Data Loaders Ready] Classes detected: {list(train_gen.class_indices.keys())}")
    return train_gen, val_gen, test_gen


def create_tf_data_pipelines(
    dataset_dir: str,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    model_name: str = "ResNet50"
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Creates high-performance tf.data pipelines with prefetching and GPU parallel loading.

    Args:
        dataset_dir (str): Root directory containing 'train', 'val', 'test'.
        target_size (Tuple[int, int]): Image size (height, width).
        batch_size (int): Batch size.
        model_name (str): Model name for preprocessing.

    Returns:
        Tuple: (train_ds, val_ds, test_ds, class_names)
    """
    AUTOTUNE = tf.data.AUTOTUNE
    preprocess_fn = get_model_preprocessing_fn(model_name)

    train_path = os.path.join(dataset_dir, "train")
    val_path = os.path.join(dataset_dir, "val")
    test_path = os.path.join(dataset_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=target_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )
    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_path,
        image_size=target_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size=target_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )

    # Augmentation layers for tf.data
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomFlip("horizontal"),
    ], name="data_augmentation")

    def preprocess_image(image, label, is_training=False):
        if is_training:
            image = data_augmentation(image, training=True)
        if preprocess_fn is not None:
            image = preprocess_fn(image)
        else:
            image = image / 255.0
        return image, label

    train_ds = train_ds.map(lambda x, y: preprocess_image(x, y, is_training=True), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = val_ds.map(lambda x, y: preprocess_image(x, y, is_training=False), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.map(lambda x, y: preprocess_image(x, y, is_training=False), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
