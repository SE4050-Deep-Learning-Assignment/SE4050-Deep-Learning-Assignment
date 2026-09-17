"""
Gradient-weighted Class Activation Mapping (Grad-CAM) engine for Explainable AI (XAI)
in Brain Tumor MRI classification. Visualizes convolutional activation heatmaps
to verify pathological feature localization.
"""

import os
from typing import List, Optional, Tuple, Any
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
try:
    import tensorflow as tf  # type: ignore
except ImportError:
    tf = None


def find_last_conv_layer(model: Any) -> str:
    """
    Automatically detects the name of the final convolutional layer in the model hierarchy.

    Args:
        model (tf.keras.Model): Model instance.

    Returns:
        str: Layer name.
    """
    for layer in reversed(model.layers):
        # If layer is a sub-model (like ResNet50 base), search inside it
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    # Known standard defaults
    name = model.name.lower()
    if "resnet" in name:
        return "conv5_block3_out"
    elif "vgg" in name:
        return "block5_conv3"
    elif "efficientnet" in name:
        return "top_conv"
    return "conv4_2"


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: Any,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Computes Grad-CAM heatmap for a single input image and target class index.

    Args:
        img_array (np.ndarray): Preprocessed input tensor shape (1, H, W, C).
        model (tf.keras.Model): Full trained model.
        last_conv_layer_name (Optional[str]): Conv layer name.
        pred_index (Optional[int]): Target class index (defaults to highest probability class).

    Returns:
        np.ndarray: 2D heatmap normalized to [0, 1].
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Check if last conv layer is inside a nested base model
    base_model = None
    target_layer = None
    for layer in model.layers:
        if hasattr(layer, "layers"):
            for sub_layer in layer.layers:
                if sub_layer.name == last_conv_layer_name:
                    base_model = layer
                    target_layer = sub_layer
                    break
        elif layer.name == last_conv_layer_name:
            target_layer = layer
            break

    if base_model is not None:
        # Build Grad-CAM model with nested base
        grad_model = tf.keras.models.Model(
            inputs=[base_model.inputs],
            outputs=[target_layer.output, base_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, base_output = grad_model(img_array)
            # Pass base_output through the remaining classifier layers of top model
            x = base_output
            for layer in model.layers[model.layers.index(base_model) + 1:]:
                x = layer(x)
            preds = x

            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
    else:
        # Standard sequential or non-nested model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        # Fallback if gradient tape did not track
        return np.ones((img_array.shape[1], img_array.shape[2]), dtype=np.float32)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight convolution feature maps by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain only features with positive influence on class
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap onto an original RGB image.

    Args:
        img_rgb (np.ndarray): Original image in uint8 format (H, W, 3).
        heatmap (np.ndarray): 2D Grad-CAM heatmap (0 to 1).
        alpha (float): Transparency factor for heatmap overlay.
        colormap (int): OpenCV colormap (default JET).

    Returns:
        np.ndarray: Blended RGB image.
    """
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Colorize
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Superimpose
    superimposed_img = heatmap_color * alpha + img_rgb * (1.0 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img


def visualize_gradcam_batch(
    model: Any,
    images: List[np.ndarray],
    true_labels: List[str],
    class_names: List[str],
    last_conv_layer_name: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Generates a clinical explanation grid displaying Original MRI, Grad-CAM Heatmap,
    and Superimposed Overlay side-by-side for a batch of sample scans.

    Args:
        model (tf.keras.Model): Trained deep learning model.
        images (List[np.ndarray]): List of raw RGB image arrays.
        true_labels (List[str]): Ground truth class labels.
        class_names (List[str]): All class categories.
        last_conv_layer_name (Optional[str]): Target conv layer.
        save_path (Optional[str]): Filepath to save figure.
    """
    n_samples = len(images)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples), dpi=150)
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (img, true_lbl) in enumerate(zip(images, true_labels)):
        # Prepare input tensor
        img_resized = cv2.resize(img, (224, 224))
        img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32)

        # Model prediction
        preds = model.predict(img_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_lbl = class_names[pred_idx]
        confidence = preds[pred_idx] * 100

        # Generate Heatmap
        heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name, pred_index=pred_idx)
        overlay = overlay_gradcam(img_resized, heatmap, alpha=0.45)

        # 1. Original
        axes[i, 0].imshow(img_resized)
        axes[i, 0].set_title(f"Original Scan\nTrue: {true_lbl}", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # 2. Heatmap
        axes[i, 1].imshow(heatmap, cmap="viridis")
        axes[i, 1].set_title(f"Grad-CAM Activation Map\nLayer: {last_conv_layer_name or 'Final Conv'}", fontsize=11)
        axes[i, 1].axis("off")

        # 3. Superimposed
        axes[i, 2].imshow(overlay)
        color = "green" if true_lbl == pred_lbl else "red"
        axes[i, 2].set_title(f"Pred: {pred_lbl} ({confidence:.1f}%)\n[Localized Pathology]", fontsize=11, color=color, fontweight="bold")
        axes[i, 2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Grad-CAM Saved] Explainability visualization saved to '{save_path}'")
    plt.show()
