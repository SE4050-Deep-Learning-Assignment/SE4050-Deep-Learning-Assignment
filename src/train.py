"""
Training engine and callback orchestration for Brain Tumor MRI classification.
Implements two-stage progressive transfer learning with learning rate adaptation.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.callbacks import (  # type: ignore
        CSVLogger,
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
        TensorBoard
    )
except ImportError:
    tf = None
    class _DummyCallback:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
    CSVLogger = EarlyStopping = ModelCheckpoint = ReduceLROnPlateau = TensorBoard = _DummyCallback


def get_callbacks(
    model_name: str,
    output_dir: str = "models",
    log_dir: str = "logs",
    patience: int = 8,
    reduce_lr_patience: int = 3,
    min_lr: float = 1e-7
) -> List[Any]:
    """
    Constructs an industrial-grade callback suite for training stability and best checkpoint retention.

    Args:
        model_name (str): Identifier name for files.
        output_dir (str): Directory where best weights/models are saved.
        log_dir (str): Directory for CSV logs and TensorBoard runs.
        patience (int): Early stopping epoch patience.
        reduce_lr_patience (int): Learning rate plateau patience.
        min_lr (float): Floor for dynamic learning rate adjustments.

    Returns:
        List[tf.keras.callbacks.Callback]: Configured callback list.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, f"{model_name}_best.keras")
    csv_log_path = os.path.join(log_dir, f"{model_name}_training_log.csv")
    tb_log_path = os.path.join(log_dir, f"tensorboard_{model_name}")

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            mode="min",
            verbose=1
        ),
        CSVLogger(csv_log_path, append=False),
        TensorBoard(log_dir=tb_log_path, histogram_freq=1)
    ]
    return callbacks


def combine_histories(h1: Any, h2: Any) -> Dict[str, List[float]]:
    """
    Merges history metrics from Phase 1 (Feature Extraction) and Phase 2 (Fine-Tuning)
    into a continuous training trajectory for plotting and analysis.

    Args:
        h1: History object or dict from Phase 1.
        h2: History object or dict from Phase 2.

    Returns:
        Dict[str, List[float]]: Unified metric trajectories.
    """
    d1 = h1.history if hasattr(h1, "history") else h1
    d2 = h2.history if hasattr(h2, "history") else h2

    combined = {}
    for key in d1.keys():
        if key in d2:
            combined[key] = list(d1[key]) + list(d2[key])
        else:
            combined[key] = list(d1[key])
    return combined


def train_two_phase_model(
    model: Any,
    base_model: Any,
    train_gen: Any,
    val_gen: Any,
    model_name: str = "ResNet50",
    stage1_epochs: int = 15,
    stage2_epochs: int = 25,
    stage1_lr: float = 1e-3,
    stage2_lr: float = 1e-5,
    unfreeze_from_layer: Optional[str] = "conv5_block1_out",
    class_weights: Optional[Dict[int, float]] = None,
    output_dir: str = "models",
    log_dir: str = "logs"
) -> Tuple[Any, Dict[str, List[float]]]:
    """
    Executes an end-to-end two-phase transfer learning workflow:
      Phase 1 (Warmup / Feature Extraction): Train custom classifier head while base is frozen.
      Phase 2 (Fine-Tuning): Unfreeze deep residual/conv layers and train at lower learning rate.

    Args:
        model (tf.keras.Model): Complete composite model.
        base_model (tf.keras.Model): Pretrained backbone.
        train_gen: Training data generator / tf.data.Dataset.
        val_gen: Validation data generator / tf.data.Dataset.
        model_name (str): Identifier name.
        stage1_epochs (int): Epochs for Stage 1.
        stage2_epochs (int): Epochs for Stage 2.
        stage1_lr (float): Initial learning rate for Stage 1.
        stage2_lr (float): Reduced learning rate for Fine-Tuning.
        unfreeze_from_layer (Optional[str]): Backbone layer name to start unfreezing.
        class_weights (Optional[Dict[int, float]]): Loss weight dictionary.
        output_dir (str): Checkpoint destination.
        log_dir (str): Log directory.

    Returns:
        Tuple[tf.keras.Model, Dict[str, List[float]]]: (trained_model, combined_history)
    """
    print(f"\n==================================================================")
    print(f"   STARTING PHASE 1: FEATURE EXTRACTION FOR {model_name.upper()}")
    print(f"   (Backbone Frozen, Initial LR = {stage1_lr})")
    print(f"==================================================================")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=stage1_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    callbacks_stage1 = get_callbacks(
        model_name=f"{model_name}_stage1",
        output_dir=output_dir,
        log_dir=log_dir,
        patience=6,
        reduce_lr_patience=2
    )

    history_stage1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=stage1_epochs,
        callbacks=callbacks_stage1,
        class_weight=class_weights,
        verbose=1
    )

    print(f"\n==================================================================")
    print(f"   STARTING PHASE 2: FINE-TUNING FOR {model_name.upper()}")
    print(f"   (Unfreezing deep layers from '{unfreeze_from_layer}', Fine-Tuning LR = {stage2_lr})")
    print(f"==================================================================")

    from src.models import unfreeze_base_model_layers
    unfreeze_base_model_layers(base_model, unfreeze_from_layer_name=unfreeze_from_layer)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=stage2_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    callbacks_stage2 = get_callbacks(
        model_name=f"{model_name}_final",
        output_dir=output_dir,
        log_dir=log_dir,
        patience=8,
        reduce_lr_patience=3
    )

    history_stage2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=stage2_epochs,
        callbacks=callbacks_stage2,
        class_weight=class_weights,
        verbose=1
    )

    combined_history = combine_histories(history_stage1, history_stage2)

    # Save final unified model
    final_model_path = os.path.join(output_dir, f"{model_name}_final.keras")
    model.save(final_model_path)
    print(f"\n[Saved] Final trained {model_name} model saved to '{final_model_path}'")

    return model, combined_history
