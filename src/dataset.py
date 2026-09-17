"""
Dataset ingestion, OpenCV cranial contour extraction, stratified splitting,
and metadata generation for the Brain Tumor MRI classification project.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


def crop_brain_contour(image: np.ndarray, plot: bool = False) -> np.ndarray:
    """
    Extracts the extreme points of the largest cranial contour in a brain MRI scan
    to crop out non-informative black borders and background artifacts.

    This is a standard medical deep learning technique (Bhuvaji et al., 2020)
    that forces CNN filters to focus on cranial parenchyma rather than padding.

    Args:
        image (np.ndarray): Input RGB or Grayscale image array (uint8).
        plot (bool): If True, plots intermediate morphological steps.

    Returns:
        np.ndarray: Cropped brain MRI scan.
    """
    # Convert to grayscale if 3 channels
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian blur to reduce high-frequency scanner noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image (binary thresholding)
    # Background in MRI is near 0, brain tissue is > 45
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)

    # Morphological erosion followed by dilation to close small gaps and remove isolated noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Find external contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback if thresholding did not find contour (e.g. extremely dark image)
        return image

    # Select the largest contour (the cranial anatomy)
    c = max(contours, key=cv2.contourArea)

    # Find the extreme coordinates
    ext_left = tuple(c[c[:, :, 0].argmin()][0])
    ext_right = tuple(c[c[:, :, 0].argmax()][0])
    ext_top = tuple(c[c[:, :, 1].argmin()][0])
    ext_bot = tuple(c[c[:, :, 1].argmax()][0])

    # Crop with safety bounding checks
    ymin = max(0, ext_top[1])
    ymax = min(image.shape[0], ext_bot[1])
    xmin = max(0, ext_left[0])
    xmax = min(image.shape[1], ext_right[0])

    # Verify valid crop dimensions
    if (ymax - ymin > 10) and (xmax - xmin > 10):
        cropped = image[ymin:ymax, xmin:xmax]
        return cropped

    return image


def scan_dataset_directory(raw_dir: str, target_classes: List[str]) -> pd.DataFrame:
    """
    Recursively scans the raw dataset directory to locate all MRI scans
    and build a structured Pandas DataFrame of image filepaths and ground-truth labels.

    Supports both standard Kaggle split layouts ('Training'/'Testing')
    and flat class folder layouts.

    Args:
        raw_dir (str): Path to raw dataset root directory.
        target_classes (List[str]): Expected class folder names.

    Returns:
        pd.DataFrame: DataFrame with columns ['filepath', 'filename', 'class_name', 'subset'].
    """
    records = []
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset path '{raw_dir}' does not exist.")

    valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}

    # Search for all image files
    for filepath in raw_path.rglob("*"):
        if filepath.suffix.lower() in valid_extensions:
            # Determine class name by inspecting path parents
            class_name = None
            for part in filepath.parts:
                normalized_part = part.lower().strip().replace(" ", "_")
                for c in target_classes:
                    if normalized_part == c or (c in normalized_part):
                        class_name = c
                        break
                if class_name:
                    break

            if class_name:
                subset = "original"
                for p in filepath.parts:
                    if p.lower() in ["training", "train"]:
                        subset = "original_train"
                    elif p.lower() in ["testing", "test", "val", "validation"]:
                        subset = "original_test"

                records.append({
                    "filepath": str(filepath.resolve()),
                    "filename": filepath.name,
                    "class_name": class_name,
                    "original_subset": subset
                })

    df = pd.DataFrame(records)
    print(f"[Dataset Scan] Found {len(df)} total images across {df['class_name'].nunique() if not df.empty else 0} classes.")
    return df


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> pd.DataFrame:
    """
    Combines the dataset pool and partitions it into stratified Train, Validation, and Test sets
    to guarantee identical class distributions across subsets and eliminate test leakage.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'filepath' and 'class_name'.
        train_ratio (float): Proportion for training (e.g. 0.70).
        val_ratio (float): Proportion for validation (e.g. 0.15).
        test_ratio (float): Proportion for test (e.g. 0.15).
        seed (int): Random seed.

    Returns:
        pd.DataFrame: Augmented DataFrame with a new 'split' column ('train', 'val', 'test').
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    df = df.copy().reset_index(drop=True)
    df["split"] = None

    # Step 1: Split into Train (train_ratio) vs Temp (val_ratio + test_ratio)
    temp_ratio = val_ratio + test_ratio
    sss_initial = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)

    for train_idx, temp_idx in sss_initial.split(df, df["class_name"]):
        df.loc[train_idx, "split"] = "train"
        temp_df = df.loc[temp_idx].copy()

    # Step 2: Split Temp into Validation and Test proportionally
    val_rel_ratio = val_ratio / temp_ratio
    sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_rel_ratio), random_state=seed)

    for val_sub_idx, test_sub_idx in sss_val_test.split(temp_df, temp_df["class_name"]):
        real_val_idx = temp_df.iloc[val_sub_idx].index
        real_test_idx = temp_df.iloc[test_sub_idx].index
        df.loc[real_val_idx, "split"] = "val"
        df.loc[real_test_idx, "split"] = "test"

    print("\n[Split Summary] Dataset Partition Distribution:")
    split_counts = pd.crosstab(df["split"], df["class_name"], margins=True)
    print(split_counts)

    return df


def calculate_class_weights(train_df: pd.DataFrame, class_list: List[str]) -> Dict[int, float]:
    """
    Calculates balanced class weights using scikit-learn for handling minor class imbalances.

    Args:
        train_df (pd.DataFrame): DataFrame with 'class_name' for training samples.
        class_list (List[str]): Ordered list of class names.

    Returns:
        Dict[int, float]: Mapping from class index (int) to weight (float).
    """
    class_to_idx = {cls: idx for idx, cls in enumerate(class_list)}
    y_train = train_df["class_name"].map(class_to_idx).values

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    weight_dict = {int(idx): float(w) for idx, w in zip(np.unique(y_train), weights)}
    print(f"\n[Class Weights] Computed balanced weights: {weight_dict}")
    for cls_name, idx in class_to_idx.items():
        print(f"  - Class '{cls_name}' (idx {idx}): weight = {weight_dict.get(idx, 1.0):.4f}")
    return weight_dict


def preprocess_and_export_dataset(
    split_df: pd.DataFrame,
    output_dir: str,
    target_size: Tuple[int, int] = (224, 224),
    crop_contour: bool = True,
    apply_clahe: bool = False,
    class_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Applies extreme-point contour cropping, optional CLAHE contrast enhancement,
    standardized resizing, and exports processed images into an organized directory structure.

    Directory Layout:
        output_dir/
            train/
                glioma_tumor/
                meningioma_tumor/
                ...
            val/
                ...
            test/
                ...
            dataset_metadata.json

    Args:
        split_df (pd.DataFrame): DataFrame with 'filepath', 'class_name', 'split'.
        output_dir (str): Root destination path for processed data.
        target_size (Tuple[int, int]): Standardized image resolution (width, height).
        crop_contour (bool): Whether to perform extreme-point contour cropping.
        apply_clahe (bool): Whether to apply CLAHE contrast enhancement.
        class_list (Optional[List[str]]): List of target classes.

    Returns:
        Dict[str, Any]: Metadata dictionary.
    """
    out_path = Path(output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if class_list is None:
        class_list = sorted(split_df["class_name"].unique().tolist())

    class_to_idx = {cls: idx for idx, cls in enumerate(class_list)}

    # Create subdirectories
    for split in ["train", "val", "test"]:
        for cls in class_list:
            (out_path / split / cls).mkdir(parents=True, exist_ok=True)

    clahe_enhancer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if apply_clahe else None

    processed_records = []
    print(f"\n[Preprocessing] Processing {len(split_df)} MRI scans to {target_size} (Cropping={crop_contour}, CLAHE={apply_clahe})...")

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Exporting Preprocessed MRI"):
        img_bgr = cv2.imread(row["filepath"])
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Extreme Point Contour Cropping
        if crop_contour:
            img_processed = crop_brain_contour(img_rgb)
        else:
            img_processed = img_rgb

        # 2. Contrast enhancement (CLAHE) if requested
        if apply_clahe:
            lab = cv2.cvtColor(img_processed, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = clahe_enhancer.apply(l_channel)
            lab = cv2.merge((l_channel, a_channel, b_channel))
            img_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 3. High-Quality Bilinear Resizing
        img_resized = cv2.resize(img_processed, target_size, interpolation=cv2.INTER_AREA)

        # 4. Save to destination
        dest_filename = f"{row['class_name']}_{Path(row['filepath']).stem}.png"
        dest_path = out_path / row["split"] / row["class_name"] / dest_filename

        # Convert RGB back to BGR for cv2.imwrite
        cv2.imwrite(str(dest_path), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

        processed_records.append({
            "processed_path": str(dest_path),
            "original_path": row["filepath"],
            "class_name": row["class_name"],
            "class_idx": class_to_idx[row["class_name"]],
            "split": row["split"]
        })

    train_df = split_df[split_df["split"] == "train"]
    class_weights = calculate_class_weights(train_df, class_list)

    metadata = {
        "classes": class_list,
        "class_to_idx": class_to_idx,
        "num_classes": len(class_list),
        "target_size": list(target_size),
        "total_images": len(processed_records),
        "split_counts": split_df["split"].value_counts().to_dict(),
        "class_weights": class_weights,
        "preprocessing_config": {
            "crop_contour": crop_contour,
            "apply_clahe": apply_clahe,
            "target_size": list(target_size)
        }
    }

    metadata_file = out_path / "dataset_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[Complete] Preprocessed dataset and metadata exported to '{output_dir}'.")
    return metadata
