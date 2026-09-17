"""
Comprehensive evaluation engine for Brain Tumor MRI classification models.
Calculates clinical-grade metrics: Accuracy, Precision, Recall/Sensitivity, Specificity,
F1-Score, ROC-AUC, and Computational Complexity (Parameters, Size, Latency).
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
try:
    import tensorflow as tf  # type: ignore
except ImportError:
    tf = None
from src.utils import plot_confusion_matrix_heatmap, plot_multiclass_roc_curves


def compute_class_specificity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[int, float]:
    """
    Computes diagnostic specificity (True Negative Rate) for each individual class in a multiclass setting.

    Args:
        y_true (np.ndarray): 1D array of ground truth class indices.
        y_pred (np.ndarray): 1D array of predicted class indices.
        num_classes (int): Total number of classes.

    Returns:
        Dict[int, float]: Specificity score per class index.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = {}
    for i in range(num_classes):
        # TN = sum of all elements except row i and column i
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        # FP = sum of column i minus the diagonal element cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities[i] = float(spec)
    return specificities


def benchmark_inference_latency(
    model: tf.keras.Model,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_iterations: int = 100,
    warmup_iterations: int = 20
) -> float:
    """
    Measures the average single-sample inference latency in milliseconds (ms).

    Args:
        model (tf.keras.Model): Model to benchmark.
        input_shape (Tuple[int, int, int]): Input shape.
        num_iterations (int): Benchmark runs.
        warmup_iterations (int): Initial runs to warm up GPU/CPU caches.

    Returns:
        float: Average latency in milliseconds.
    """
    dummy_input = np.random.randn(1, *input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup_iterations):
        _ = model(dummy_input, training=False)

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = model(dummy_input, training=False)
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / num_iterations) * 1000.0
    return float(avg_latency_ms)


def evaluate_model_comprehensive(
    model: tf.keras.Model,
    test_gen: Any,
    class_names: List[str],
    model_name: str = "ResNet50",
    results_dir: str = "results",
    input_shape: Tuple[int, int, int] = (224, 224, 3)
) -> Dict[str, Any]:
    """
    Performs full multi-metric evaluation of a trained model on the unseen test dataset.

    Args:
        model (tf.keras.Model): The trained model.
        test_gen: Test data generator (must have shuffle=False).
        class_names (List[str]): Ordered list of diagnostic class names.
        model_name (str): Model identifier.
        results_dir (str): Output folder for metrics and plots.
        input_shape (Tuple[int, int, int]): Input dimensions.

    Returns:
        Dict[str, Any]: Comprehensive evaluation metrics dictionary.
    """
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n==================================================================")
    print(f"   COMPREHENSIVE TEST SET EVALUATION: {model_name.upper()}")
    print(f"==================================================================")

    # Reset generator
    if hasattr(test_gen, "reset"):
        test_gen.reset()

    # Predictions
    print("[Evaluation] Generating predictions on unseen test dataset...")
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    if hasattr(test_gen, "classes"):
        y_true = test_gen.classes
    elif hasattr(test_gen, "labels"):
        y_true = test_gen.labels
    else:
        # If tf.data dataset, extract true labels
        y_true_list = []
        for _, labels in test_gen:
            y_true_list.append(labels.numpy())
        y_true_arr = np.concatenate(y_true_list, axis=0)
        y_true = np.argmax(y_true_arr, axis=1) if len(y_true_arr.shape) > 1 else y_true_arr

    num_classes = len(class_names)
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)

    # 1. Performance Metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Specificity
    specificities = compute_class_specificity(y_true, y_pred, num_classes)
    spec_macro = float(np.mean(list(specificities.values())))

    # Multi-class ROC-AUC
    try:
        auc_macro = float(roc_auc_score(y_true_onehot, y_pred_proba, average="macro", multi_class="ovr"))
        auc_weighted = float(roc_auc_score(y_true_onehot, y_pred_proba, average="weighted", multi_class="ovr"))
    except Exception as e:
        print(f"[AUC Warning] Could not calculate sklearn AUC: {e}")
        auc_macro = 0.0
        auc_weighted = 0.0

    # 2. Model Complexity & Operational Metrics
    total_params = int(model.count_params())
    trainable_params = int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    non_trainable_params = int(total_params - trainable_params)
    latency_ms = benchmark_inference_latency(model, input_shape=input_shape)

    # Estimate model size in MB
    model_size_mb = (total_params * 4.0) / (1024 * 1024)

    # 3. Formatted Classification Report
    clf_report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    clf_report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\n[Diagnostic Performance Summary]")
    print(f"  • Overall Test Accuracy:   {acc*100:.2f}%")
    print(f"  • Macro F1-Score:          {f1_macro:.4f}")
    print(f"  • Weighted F1-Score:       {f1_weighted:.4f}")
    print(f"  • Macro Precision:         {prec_macro:.4f}")
    print(f"  • Macro Recall/Sensitivity:{rec_macro:.4f}")
    print(f"  • Macro Specificity:       {spec_macro:.4f}")
    print(f"  • Multi-Class ROC-AUC:     {auc_macro:.4f}")
    print("\n[Computational Complexity]")
    print(f"  • Total Parameters:        {total_params:,}")
    print(f"  • Trainable Parameters:    {trainable_params:,}")
    print(f"  • Model Size:              {model_size_mb:.2f} MB")
    print(f"  • Inference Latency:       {latency_ms:.2f} ms/sample")
    print("\n[Detailed Classification Report]")
    print(clf_report_text)

    # 4. Generate Visualizations
    cm_plot_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix_heatmap(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        model_name=model_name,
        save_path=cm_plot_path
    )

    roc_plot_path = os.path.join(results_dir, f"{model_name}_roc_curves.png")
    per_class_auc = plot_multiclass_roc_curves(
        y_true_onehot=y_true_onehot,
        y_pred_proba=y_pred_proba,
        class_names=class_names,
        model_name=model_name,
        save_path=roc_plot_path
    )

    results_dict = {
        "model_name": model_name,
        "test_accuracy": acc,
        "macro_precision": prec_macro,
        "weighted_precision": prec_weighted,
        "macro_recall": rec_macro,
        "weighted_recall": rec_weighted,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "macro_specificity": spec_macro,
        "macro_roc_auc": auc_macro,
        "weighted_roc_auc": auc_weighted,
        "per_class_specificity": {class_names[k]: v for k, v in specificities.items()},
        "per_class_auc": per_class_auc,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "model_size_mb": model_size_mb,
        "inference_latency_ms": latency_ms,
        "classification_report": clf_report
    }

    # Save to JSON
    json_path = os.path.join(results_dir, f"{model_name}_evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"[Results Saved] Detailed metrics JSON saved to '{json_path}'")

    return results_dict
