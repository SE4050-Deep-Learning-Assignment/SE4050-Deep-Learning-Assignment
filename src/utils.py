"""
Utility functions for reproducibility, hardware acceleration detection,
and scientific publication-grade visualizations for Brain Tumor MRI Deep Learning.
"""

import os
import random
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
try:
    import tensorflow as tf  # type: ignore
except ImportError:
    tf = None


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across Python, NumPy, and TensorFlow for strict reproducibility.

    Args:
        seed (int): The integer seed value. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Ensure deterministic GPU operations where supported
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    print(f"[Info] Reproducibility seed set to {seed}")


def check_gpu() -> Dict[str, Any]:
    """
    Checks for available GPU devices and prints accelerator diagnostics.

    Returns:
        Dict[str, Any]: Dictionary containing device name and GPU availability.
    """
    gpus = tf.config.list_physical_devices("GPU")
    gpu_available = len(gpus) > 0
    device_info = {
        "gpu_available": gpu_available,
        "device_count": len(gpus),
        "devices": [gpu.name for gpu in gpus]
    }
    if gpu_available:
        print(f"[Hardware] GPU detected! Active accelerator: {gpus[0].name}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"[Hardware] Enabled dynamic memory growth for {gpu.name}")
            except RuntimeError as e:
                print(f"[Hardware Warning] Could not set memory growth: {e}")
    else:
        print("[Hardware] No GPU detected. Running on CPU (training may be slow).")
    return device_info


def plot_learning_curves(
    history: Any,
    model_name: str = "ResNet50",
    save_path: Optional[str] = None
) -> None:
    """
    Plots professional training and validation curves for Loss, Accuracy, and AUC.

    Args:
        history: Keras History object or history dictionary.
        model_name (str): Name of the model for title formatting.
        save_path (Optional[str]): Path to save the figure image.
    """
    hist = history.history if hasattr(history, "history") else history
    epochs = range(1, len(hist["loss"]) + 1)

    has_auc = "auc" in hist or "auc_1" in hist or any("auc" in k.lower() for k in hist.keys())
    ncols = 3 if has_auc else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), dpi=150)

    # Style configuration
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Loss Curve
    axes[0].plot(epochs, hist["loss"], label="Train Loss", color="#1f77b4", lw=2)
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], label="Val Loss", color="#d62728", lw=2, linestyle="--")
    axes[0].set_title(f"{model_name} - Cross-Entropy Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epochs", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True)

    # 2. Accuracy Curve
    acc_key = "accuracy" if "accuracy" in hist else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in hist else "val_acc"
    if acc_key in hist:
        axes[1].plot(epochs, hist[acc_key], label="Train Accuracy", color="#2ca02c", lw=2)
        if val_acc_key in hist:
            axes[1].plot(epochs, hist[val_acc_key], label="Val Accuracy", color="#ff7f0e", lw=2, linestyle="--")
        axes[1].set_title(f"{model_name} - Classification Accuracy", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epochs", fontsize=11)
        axes[1].set_ylabel("Accuracy", fontsize=11)
        axes[1].set_ylim([0.0, 1.05])
        axes[1].legend(loc="lower right", frameon=True)

    # 3. AUC Curve (if recorded)
    if has_auc:
        auc_key = next((k for k in hist if "auc" in k.lower() and not k.startswith("val_")), None)
        val_auc_key = next((k for k in hist if "auc" in k.lower() and k.startswith("val_")), None)
        if auc_key and auc_key in hist:
            axes[2].plot(epochs, hist[auc_key], label="Train AUC", color="#9467bd", lw=2)
            if val_auc_key and val_auc_key in hist:
                axes[2].plot(epochs, hist[val_auc_key], label="Val AUC", color="#8c564b", lw=2, linestyle="--")
            axes[2].set_title(f"{model_name} - Area Under ROC Curve (AUC)", fontsize=13, fontweight="bold")
            axes[2].set_xlabel("Epochs", fontsize=11)
            axes[2].set_ylabel("AUC Score", fontsize=11)
            axes[2].set_ylim([0.0, 1.05])
            axes[2].legend(loc="lower right", frameon=True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot Saved] Learning curves saved to '{save_path}'")
    plt.show()


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str = "ResNet50",
    save_path: Optional[str] = None
) -> None:
    """
    Renders an annotated normalized confusion matrix heatmap for clinical error analysis.

    Args:
        y_true (np.ndarray): Ground truth class indices.
        y_pred (np.ndarray): Predicted class indices.
        class_names (List[str]): List of human-readable class names.
        model_name (str): Name of the model.
        save_path (Optional[str]): Filepath to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Combine counts and percentages into labels
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    plt.figure(figsize=(8, 7), dpi=150)
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=1,
        linecolor="white"
    )
    plt.title(f"{model_name} - Test Set Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Diagnostic Class", fontsize=12, labelpad=10)
    plt.ylabel("True Diagnostic Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot Saved] Confusion matrix saved to '{save_path}'")
    plt.show()


def plot_multiclass_roc_curves(
    y_true_onehot: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    model_name: str = "ResNet50",
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Computes and plots One-vs-Rest ROC curves for each tumor class with Micro/Macro averages.

    Args:
        y_true_onehot (np.ndarray): One-hot encoded ground truth labels (N, num_classes).
        y_pred_proba (np.ndarray): Predicted class probabilities (N, num_classes).
        class_names (List[str]): List of class names.
        model_name (str): Name of the model.
        save_path (Optional[str]): Filepath to save the plot.

    Returns:
        Dict[str, float]: Dictionary of AUC scores per class and macro/micro average.
    """
    n_classes = len(class_names)
    fpr: Dict[Any, np.ndarray] = {}
    tpr: Dict[Any, np.ndarray] = {}
    roc_auc: Dict[str, float] = {}

    # Per-class ROC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        roc_auc[class_names[i]] = float(auc(fpr[i], tpr[i]))

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_pred_proba.ravel())
    roc_auc["micro_avg"] = float(auc(fpr["micro"], tpr["micro"]))

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro_avg"] = float(auc(fpr["macro"], tpr["macro"]))

    # Plot
    plt.figure(figsize=(9, 7), dpi=150)
    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f"Micro-Average ROC (AUC = {roc_auc['micro_avg']:.4f})",
        color="deeppink", linestyle=":", linewidth=2.5
    )
    plt.plot(
        fpr["macro"], tpr["macro"],
        label=f"Macro-Average ROC (AUC = {roc_auc['macro_avg']:.4f})",
        color="navy", linestyle=":", linewidth=2.5
    )

    colors = ["#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd"]
    for i in range(n_classes):
        plt.plot(
            fpr[i], tpr[i],
            color=colors[i % len(colors)],
            lw=2,
            label=f"ROC: {class_names[i]} (AUC = {roc_auc[class_names[i]]:.4f})"
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.50)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title(f"{model_name} - Multi-Class ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot Saved] ROC curves saved to '{save_path}'")
    plt.show()

    return roc_auc


def plot_sample_mri_grid(
    images: List[np.ndarray],
    labels: List[str],
    predictions: Optional[List[str]] = None,
    grid_size: Tuple[int, int] = (2, 4),
    title: str = "MRI Scans Sample Inspection",
    save_path: Optional[str] = None
) -> None:
    """
    Renders an inspection grid of MRI images with their ground truth and optional model predictions.

    Args:
        images (List[np.ndarray]): List of image arrays.
        labels (List[str]): Ground truth labels.
        predictions (Optional[List[str]]): Optional predicted labels.
        grid_size (Tuple[int, int]): (rows, cols).
        title (str): Figure title.
        save_path (Optional[str]): Filepath to save figure.
    """
    rows, cols = grid_size
    num_samples = min(len(images), rows * cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
    axes = np.array(axes).reshape(-1)

    for i in range(num_samples):
        img = images[i]
        # Rescale if in [0, 1] or uint8
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        axes[i].imshow(img)
        axes[i].axis("off")

        true_lbl = labels[i]
        if predictions is not None:
            pred_lbl = predictions[i]
            is_correct = (true_lbl == pred_lbl)
            color = "green" if is_correct else "red"
            axes[i].set_title(f"True: {true_lbl}\nPred: {pred_lbl}", fontsize=11, color=color, fontweight="bold")
        else:
            axes[i].set_title(f"Class: {true_lbl}", fontsize=11, fontweight="bold")

    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Plot Saved] MRI Grid saved to '{save_path}'")
    plt.show()
