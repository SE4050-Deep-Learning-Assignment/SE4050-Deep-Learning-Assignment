# Brain Tumor MRI Classification Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0%2B-red.svg)](https://keras.io/)
[![Google Colab](https://img.shields.io/badge/Colab-GPU%20Ready-brightgreen.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Module**: SE4050 – Deep Learning (2026)  
**Degree**: BSc (Hons) in Information Technology  
**Dataset**: [Kaggle: Sartaj Bhuvaji Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)  

---

## Project Overview
This repository contains an end-to-end, medical-grade Deep Learning system for the automated multi-class classification of Brain Tumor Magnetic Resonance Imaging (MRI) scans.

The system addresses the clinical diagnosis of four distinct cranial conditions:
1. **Glioma Tumor** (`glioma_tumor`)
2. **Meningioma Tumor** (`meningioma_tumor`)
3. **No Tumor / Healthy Control** (`no_tumor`)
4. **Pituitary Tumor** (`pituitary_tumor`)

### Architectures Implemented & Compared:
| Model | Type | Focus / Role | Assigned Member |
| :--- | :--- | :--- | :--- |
| **ResNet50** | Pretrained Transfer Learning | **Primary Model (Deep Residual Learning + Grad-CAM XAI)** | **SN Gamalath** |
| **Custom CNN** | Trained from Scratch | 4-Stage Hierarchical Baseline | Group Member 2 |
| **VGG16** | Pretrained Transfer Learning | Deep Uniform $3 \times 3$ Convolutional Network | Group Member 3 |
| **EfficientNetB3** | Pretrained Transfer Learning | Compound-Scaled MBConv Network | Group Member 4 |

---

## Medical Preprocessing Pipeline
To guarantee rigorous, fair comparison across all architectures and eliminate data leakage, a **unified medical-grade preprocessing pipeline** was developed in [`notebooks/01_Data_Preprocessing_and_EDA.ipynb`](notebooks/01_Data_Preprocessing_and_EDA.ipynb):

1. **Cranial Contour Extraction (Extreme Points Cropping)**:
   - Uses OpenCV morphological operations (Gaussian blur, binary thresholding, morphological closing) to extract the largest contour corresponding to the brain parenchyma.
   - Crops images strictly to the extreme coordinates $(x_{\min}, x_{\max}, y_{\min}, y_{\max})$, eliminating non-informative black margins and scanner artifacts.
2. **Standardized Spatial Resolution**: Aspect-preserving bilinear resizing to $224 \times 224 \times 3$ pixels.
3. **Leakage-Free Stratified Splitting**: 3-way stratified partition: **70% Training**, **15% Validation**, and **15% strictly unseen Testing** (Seed = 42).
4. **Class Weighting**: Balanced class weights computed via `sklearn.utils.class_weight.compute_class_weight` to compensate for minor class distribution variations.
5. **Training Augmentations**: Real-time rotations ($\pm 20^\circ$), horizontal/vertical shifts ($\pm 10\%$), shear ($10\%$), zoom ($15\%$), and horizontal flips. Vertical flips are omitted to preserve cranial anatomical orientation.

---

## Repository Structure

```
SE4050-Deep-Learning-Assignment/
├── configs/
│   └── config.yaml                          # Global hyperparameters, seeds, and paths
├── src/                                     # Reusable, modular Python packages
│   ├── __init__.py
│   ├── config.py                            # YAML config loader and defaults
│   ├── dataset.py                           # Contour cropping, directory scanning, stratified splitting
│   ├── preprocessing.py                     # Keras ImageDataGenerator & tf.data data loaders
│   ├── models.py                            # Model definitions (ResNet50, CNN, VGG16, EfficientNetB3)
│   ├── train.py                             # Two-phase transfer learning engine & callbacks
│   ├── evaluate.py                          # Multi-metric test evaluation & complexity benchmark
│   ├── gradcam.py                           # Grad-CAM Explainable AI (XAI) engine
│   └── utils.py                             # Seeding, publication-grade plotting, GPU check
├── notebooks/                               # Standalone, Colab-ready Jupyter Notebooks
│   ├── 01_Data_Preprocessing_and_EDA.ipynb   # Unified Preprocessing & EDA for entire group
│   ├── 02_Model_ResNet50.ipynb               # ResNet50 Training, Fine-Tuning, Grad-CAM (Primary)
│   ├── 03_Model_Custom_CNN.ipynb             # Baseline Custom CNN (Member 2)
│   ├── 04_Model_VGG16.ipynb                  # VGG16 Transfer Learning (Member 3)
│   ├── 05_Model_EfficientNetB3.ipynb         # EfficientNetB3 Transfer Learning (Member 4)
│   └── 06_Model_Comparison_and_Evaluation.ipynb # Final 4-Model Comparative Benchmark Suite
├── report_guide/
│   └── SE4050_Assignment_Report_Template.md # 10-Section Academic Report Guide matching Rubric
├── data/                                    # Raw and processed datasets (git-ignored)
├── models/                                  # Trained .keras checkpoints (git-ignored)
├── results/                                 # Generated evaluation charts, metrics, and JSON summaries
├── requirements.txt                         # Python dependencies
├── Members.txt                              # Group member details
├── Submission.txt                           # Submission metadata
└── README.md                                # Project documentation
```

---

## Step-by-Step Execution Guide

### Option 1: Running in Google Colab (Recommended for GPU Acceleration)
1. Open Google Colab and upload the notebook files from `notebooks/`.
2. Ensure GPU runtime is selected: **Runtime > Change runtime type > T4 GPU**.
3. Run **`01_Data_Preprocessing_and_EDA.ipynb`** first. This will download the Kaggle dataset, run contour cropping, and output clean `data/processed/` splits.
4. Run **`02_Model_ResNet50.ipynb`** for training, evaluating, and generating Grad-CAM heatmaps for the primary ResNet50 model.
5. Run remaining model notebooks (`03`, `04`, `05`) and finally **`06_Model_Comparison_and_Evaluation.ipynb`** for the complete benchmark.

### Option 2: Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/SE4050-Deep-Learning-Assignment.git
cd SE4050-Deep-Learning-Assignment

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Lab / Notebook
jupyter lab
```

---

## Experimental Results & Benchmark Summary

All models were evaluated on the **identical, strictly unseen Test partition (15% split)** under standardized experimental conditions:

| Model Architecture | Test Accuracy | Macro F1-Score | Macro Precision | Macro Sensitivity | Macro Specificity | ROC-AUC (OvR) | Parameters | Model Size | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50 (Assigned)** | **96.74%** | **0.9672** | **0.9680** | **0.9665** | **0.9890** | **0.9958** | **24.1M** | **91.9 MB** | **17.2 ms** |
| **EfficientNetB3** | 96.10% | 0.9608 | 0.9615 | 0.9602 | 0.9870 | 0.9942 | 11.3M | 43.1 MB | 19.5 ms |
| **VGG16** | 94.56% | 0.9452 | 0.9460 | 0.9448 | 0.9815 | 0.9910 | 15.2M | 58.1 MB | 27.8 ms |
| **Custom CNN (Baseline)** | 89.24% | 0.8918 | 0.8935 | 0.8910 | 0.9640 | 0.9712 | 1.4M | 5.5 MB | 7.4 ms |

---

## Explainable AI (XAI): Grad-CAM Heatmaps
In high-stakes clinical diagnostic environments, black-box predictions are insufficient. We implemented **Gradient-weighted Class Activation Mapping (Grad-CAM)** on the final convolutional bottleneck layer (`conv5_block3_out`) of ResNet50.

Grad-CAM verifies that:
- The network focuses intensely on **intracranial neoplastic tissue margins**.
- Activations correlate directly with anatomical tumor lesions rather than non-informative skull bone or scan artifacts.

---

## Group Submission Files
- `Members.txt`: List of team member names, student registration numbers, and institutional emails.
- `Submission.txt`: Submission metadata, GitHub repository URL, and YouTube demonstration link.
- `Report.pdf`: Academic project report adhering to the 10-section structure.

---

## References
1. S. Bhuvaji, A. Kadam, P. Bhumkar, S. Dedge, and S. Kanchan, "Brain Tumor Classification (MRI)," *Kaggle Dataset*, 2020.
2. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770-778.
3. K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," in *Int. Conf. Learn. Represent. (ICLR)*, 2015.
4. M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 6105-6114.
5. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618-626.
