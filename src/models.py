"""
Model architectural definitions and transfer learning builders for:
1. ResNet50 (Primary assigned architecture with residual connections)
2. Custom CNN (Hierarchical baseline trained from scratch)
3. VGG16 (Deep uniform 3x3 convolutional architecture)
4. EfficientNetB3 (Compound-scaled MBConv architecture)
"""

from typing import List, Optional, Tuple, Any
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models, regularizers  # type: ignore
except ImportError:
    tf = None
    class _DummyCallable:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self
        def __getattr__(self, name: str) -> Any:
            return self
    layers = models = regularizers = _DummyCallable()


def build_resnet50_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    dense_units: List[int] = [256],
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
    freeze_base: bool = True
) -> Tuple[Any, Any]:
    """
    Constructs a transfer-learning architecture based on ResNet50 with ImageNet pretrained weights
    and a custom deep regularization head.

    Args:
        input_shape (Tuple[int, int, int]): Input MRI dimensions (H, W, C).
        num_classes (int): Number of target diagnostic categories (4).
        dense_units (List[int]): Dimensions of fully-connected classification layers.
        dropout_rate (float): Dropout probability for regularization.
        l2_reg (float): L2 weight decay penalty.
        freeze_base (bool): Whether to freeze backbone weights initially.

    Returns:
        Tuple[tf.keras.Model, tf.keras.Model]: (full_model, base_model)
    """
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = not freeze_base

    inputs = layers.Input(shape=input_shape, name="mri_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)

    for i, units in enumerate(dense_units):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"head_dense_{i+1}"
        )(x)
        x = layers.Dropout(dropout_rate, name=f"head_dropout_{i+1}")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="diagnostic_output"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet50_BrainTumor")
    return model, base_model


def build_custom_cnn_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    dropout_rate: float = 0.3
) -> Any:
    """
    Constructs a 4-stage convolutional baseline architecture with Batch Normalization
    and progressive filter scaling (32 -> 64 -> 128 -> 256) trained from scratch.

    Args:
        input_shape (Tuple[int, int, int]): Dimensions of input MRI.
        num_classes (int): Number of diagnostic categories.
        dropout_rate (float): Dropout rate.

    Returns:
        Any: Compiled Custom CNN model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1: 32 Filters
        layers.Conv2D(32, (3, 3), padding="same", name="conv1_1"),
        layers.BatchNormalization(name="bn1_1"),
        layers.Activation("relu"),
        layers.Conv2D(32, (3, 3), padding="same", name="conv1_2"),
        layers.BatchNormalization(name="bn1_2"),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Dropout(0.2, name="drop1"),

        # Block 2: 64 Filters
        layers.Conv2D(64, (3, 3), padding="same", name="conv2_1"),
        layers.BatchNormalization(name="bn2_1"),
        layers.Activation("relu"),
        layers.Conv2D(64, (3, 3), padding="same", name="conv2_2"),
        layers.BatchNormalization(name="bn2_2"),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Dropout(0.25, name="drop2"),

        # Block 3: 128 Filters
        layers.Conv2D(128, (3, 3), padding="same", name="conv3_1"),
        layers.BatchNormalization(name="bn3_1"),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2), name="pool3"),
        layers.Dropout(0.3, name="drop3"),

        # Block 4: 256 Filters
        layers.Conv2D(256, (3, 3), padding="same", name="conv4_1"),
        layers.BatchNormalization(name="bn4_1"),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2), name="pool4"),
        layers.Dropout(0.35, name="drop4"),

        # Dense Classifier Head
        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dense(256, activation="relu", name="dense1"),
        layers.Dropout(dropout_rate, name="drop_dense1"),
        layers.Dense(128, activation="relu", name="dense2"),
        layers.Dropout(dropout_rate, name="drop_dense2"),
        layers.Dense(num_classes, activation="softmax", name="output")
    ], name="Custom_Hierarchical_CNN")

    return model


def build_vgg16_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    dense_units: List[int] = [512, 256],
    dropout_rate: float = 0.5,
    l2_reg: float = 1e-4,
    freeze_base: bool = True
) -> Tuple[Any, Any]:
    """
    Constructs a VGG16 transfer learning model with custom classification head.

    Args:
        input_shape (Tuple[int, int, int]): Input dimensions.
        num_classes (int): Number of diagnostic categories.
        dense_units (List[int]): Head dense units.
        dropout_rate (float): Dropout probability.
        l2_reg (float): L2 regularization factor.
        freeze_base (bool): Whether to freeze backbone.

    Returns:
        Tuple[Any, Any]: (full_model, base_model)
    """
    base_model = tf.keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = not freeze_base

    inputs = layers.Input(shape=input_shape, name="mri_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)

    for i, units in enumerate(dense_units):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"head_dense_{i+1}"
        )(x)
        x = layers.Dropout(dropout_rate, name=f"head_dropout_{i+1}")(x)

    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="output")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="VGG16_BrainTumor")
    return model, base_model


def build_efficientnet_b3_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    dense_units: List[int] = [256],
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
    freeze_base: bool = True
) -> Tuple[Any, Any]:
    """
    Constructs an EfficientNetB3 transfer learning architecture with compound scaling.

    Args:
        input_shape (Tuple[int, int, int]): Input dimensions.
        num_classes (int): Number of diagnostic categories.
        dense_units (List[int]): Head dense units.
        dropout_rate (float): Dropout probability.
        l2_reg (float): L2 regularizer coefficient.
        freeze_base (bool): Whether to freeze backbone.

    Returns:
        Tuple[Any, Any]: (full_model, base_model)
    """
    base_model = tf.keras.applications.EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = not freeze_base

    inputs = layers.Input(shape=input_shape, name="mri_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)

    for i, units in enumerate(dense_units):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"head_dense_{i+1}"
        )(x)
        x = layers.Dropout(dropout_rate, name=f"head_dropout_{i+1}")(x)

    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="output")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB3_BrainTumor")
    return model, base_model


def unfreeze_base_model_layers(
    base_model: Any,
    unfreeze_from_layer_name: Optional[str] = None,
    num_unfrozen_layers: Optional[int] = None
) -> None:
    """
    Selectively unfreezes deep convolutional layers in a pretrained backbone for Stage 2 Fine-Tuning.

    Args:
        base_model (tf.keras.Model): Backbone model instance.
        unfreeze_from_layer_name (Optional[str]): Unfreeze all layers starting from this layer name.
        num_unfrozen_layers (Optional[int]): Alternatively, unfreeze the top N layers.
    """
    base_model.trainable = True

    if unfreeze_from_layer_name:
        set_trainable = False
        for layer in base_model.layers:
            if layer.name == unfreeze_from_layer_name:
                set_trainable = True
            layer.trainable = set_trainable
            # Keep BatchNormalization layers frozen in fine-tuning to preserve ImageNet running statistics
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    elif num_unfrozen_layers is not None:
        for layer in base_model.layers[:-num_unfrozen_layers]:
            layer.trainable = False
        for layer in base_model.layers[-num_unfrozen_layers:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    trainable_count = sum([tf.keras.backend.count_params(w) for w in base_model.trainable_weights])
    total_count = sum([tf.keras.backend.count_params(w) for w in base_model.weights])
    print(f"[Fine-Tuning Setup] Base model unfrozen. Trainable parameters: {trainable_count:,} / {total_count:,}")
