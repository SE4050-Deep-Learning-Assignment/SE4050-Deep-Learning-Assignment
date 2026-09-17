"""
Configuration loader and constants for the Brain Tumor MRI classification project.
Supports loading from configs/config.yaml with robust defaults for Google Colab and local environments.
"""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "Brain-Tumor-MRI-Classification",
        "version": "1.0.0",
        "seed": 42
    },
    "dataset": {
        "kaggle_dataset_slug": "sartajbhuvaji/brain-tumor-classification-mri",
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "classes": ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"],
        "num_classes": 4,
        "image_size": [224, 224],
        "num_channels": 3,
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "stratified": True
    },
    "preprocessing": {
        "crop_contour": True,
        "apply_clahe": False,
        "normalization": "rescale_255",
        "batch_size": 32
    },
    "augmentation": {
        "enabled": True,
        "rotation_range": 20,
        "width_shift_range": 0.10,
        "height_shift_range": 0.10,
        "shear_range": 0.10,
        "zoom_range": 0.15,
        "horizontal_flip": True,
        "vertical_flip": False,
        "fill_mode": "nearest",
        "brightness_range": [0.85, 1.15]
    },
    "models": {
        "resnet50": {
            "name": "ResNet50",
            "weights": "imagenet",
            "include_top": False,
            "pooling": "avg",
            "dense_units": [256],
            "dropout_rate": 0.4,
            "l2_reg": 0.0001,
            "freeze_base_epochs": 15,
            "fine_tune_epochs": 25,
            "initial_lr": 0.001,
            "fine_tune_lr": 0.00001,
            "fine_tune_layers_from": "conv5_block1_out"
        },
        "custom_cnn": {
            "name": "Custom_CNN",
            "filters": [32, 64, 128, 256],
            "dense_units": [256, 128],
            "dropout_rate": 0.3,
            "epochs": 35,
            "initial_lr": 0.001
        },
        "vgg16": {
            "name": "VGG16",
            "weights": "imagenet",
            "include_top": False,
            "pooling": "avg",
            "dense_units": [512, 256],
            "dropout_rate": 0.5,
            "l2_reg": 0.0001,
            "freeze_base_epochs": 15,
            "fine_tune_epochs": 25,
            "initial_lr": 0.001,
            "fine_tune_lr": 0.00001,
            "fine_tune_layers_from": "block5_conv1"
        },
        "efficientnet_b3": {
            "name": "EfficientNetB3",
            "weights": "imagenet",
            "include_top": False,
            "pooling": "avg",
            "dense_units": [256],
            "dropout_rate": 0.4,
            "l2_reg": 0.0001,
            "freeze_base_epochs": 15,
            "fine_tune_epochs": 25,
            "initial_lr": 0.001,
            "fine_tune_lr": 0.00005,
            "fine_tune_layers_from": "block6a_expand_conv"
        }
    },
    "training": {
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy", "AUC"],
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 8,
            "restore_best_weights": True
        },
        "reduce_lr": {
            "monitor": "val_loss",
            "factor": 0.2,
            "patience": 3,
            "min_lr": 0.0000001
        }
    },
    "outputs": {
        "models_dir": "models",
        "logs_dir": "logs",
        "results_dir": "results"
    }
}


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load project configuration from YAML file, falling back to DEFAULT_CONFIG if not found.

    Args:
        config_path (str): Path to config YAML.

    Returns:
        Dict[str, Any]: Dictionary with configuration parameters.
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            try:
                loaded = yaml.safe_load(f)
                return loaded if loaded is not None else DEFAULT_CONFIG
            except Exception as e:
                print(f"[Warning] Failed to parse config file '{config_path}': {e}. Using defaults.")
                return DEFAULT_CONFIG
    return DEFAULT_CONFIG
