# 📄 SE4050 – Deep Learning Assignment: Comprehensive Academic Report Guide & Template

**Module**: SE4050 – Deep Learning (2026)  
**Degree**: BSc (Hons) in Information Technology  
**Project**: Brain Tumor Classification from Magnetic Resonance Imaging (MRI) Using Deep Learning Architectures  
**Category**: Supervised Deep Learning  
**Architectures Compared**: ResNet50 (Primary Assigned Model), Custom CNN (Baseline), VGG16, EfficientNetB3  

---

## 📑 Required Report Structure (10 Sections Aligned with Rubric)

### Section 1: Introduction and Problem Definition
- **Clinical Motivation**: Brain tumors (gliomas, meningiomas, pituitary tumors) represent one of the most fatal oncological conditions worldwide. Early, accurate detection and differential diagnosis using Magnetic Resonance Imaging (MRI) is vital for surgical planning, radiotherapy, and survival prognosis.
- **Problem Statement**: Manual interpretation of MRI scans by radiologists is time-intensive, subject to inter-observer variability, and prone to diagnostic delays under heavy clinical workloads.
- **Deep Learning Objective**: Develop, optimize, and critically evaluate an automated, multi-class Computer-Aided Diagnostic (CAD) system utilizing Convolutional Neural Networks (CNNs) and deep transfer learning models to classify brain MRI scans into four diagnostic categories (`glioma_tumor`, `meningioma_tumor`, `no_tumor`, `pituitary_tumor`).

---

### Section 2: Background and Related Work
- **Evolution of CNNs in Medical Neuroimaging**: Review classical handcrafted feature extractors (GLCM, Gabor filters, Wavelets with SVM) versus modern end-to-end Deep Convolutional Neural Networks.
- **Transfer Learning in Radiology**: Explain why ImageNet-pretrained representations (edge detectors, texture filters) transfer effectively to medical domains despite domain shifts, mitigating small-sample overfitting.
- **Key Literature Benchmarks**:
  - *Bhuvaji et al. (2020)*: Baseline evaluation of brain tumor classification using basic CNNs.
  - *He et al. (2016)*: Residual Learning Framework (ResNet) and identity shortcut formulations.
  - *Simonyan & Zisserman (2014)*: VGG architectures with deep stacks of $3 \times 3$ convolutions.
  - *Tan & Le (2019)*: EfficientNet compound scaling principles.
  - *Selvaraju et al. (2017)*: Grad-CAM for visual explanations and transparency in AI.

---

### Section 3: Dataset Description and Exploratory Data Analysis (EDA)
- **Dataset Provenance & Context**:
  - **Source**: Sartaj Bhuvaji Brain Tumor Classification Dataset (Kaggle).
  - **Classes (4)**:
    1. `glioma_tumor`: Infiltrative intrinsic brain tumors arising from glial cells.
    2. `meningioma_tumor`: Typically benign, extra-axial tumors originating from meninges.
    3. `no_tumor`: Healthy cranial controls without detectable neoplastic lesions.
    4. `pituitary_tumor`: Benign adenomas situated at the skull base/sella turcica.
- **Dimensional & Statistical Summary**:
  - Total Scans: $\approx 3,264$ raw scans.
  - Variable image resolutions (e.g., $512 \times 512$, $256 \times 256$, $630 \times 587$).
  - Distribution of classes and visual examination of intensity histograms.
- **Identified Data Quality Challenges**:
  - Inconsistent scanner resolutions and field-of-view.
  - Extensive black background margins containing zero diagnostic information.
  - Presence of scanner noise, patient motion artifacts, and slight class frequency variations.

---

### Section 4: Data Preprocessing and Feature Engineering
- **Unified Pipeline Architecture**:
  1. **Cranial Contour Extraction (Extreme Points Cropping)**:
     - Applied Gaussian smoothing ($5 \times 5$) and binary thresholding ($T=45$).
     - Performed morphological closing (erosion + dilation with rectangular kernel) to suppress background noise.
     - Extracted the largest contour corresponding to the intracranial parenchyma and cropped strictly to the extreme points $(x_{\min}, x_{\max}, y_{\min}, y_{\max})$.
     - *Justification*: Removes non-brain margins, forcing convolutional kernels to optimize filters on pathological lesions.
  2. **Standardized Spatial Resizing**: Bilinear interpolation resizing to $224 \times 224 \times 3$.
  3. **Input Normalization & Framework Statistics**:
     - Standard $[0, 1]$ scaling for Custom CNN.
     - Zero-centered channel subtraction (`preprocess_input`) for ResNet50, VGG16, and EfficientNetB3 matching ImageNet distribution.
  4. **Leakage-Free Stratified Partitioning**:
     - Pooled dataset partitioned into $70\%$ Training, $15\%$ Validation, and $15\%$ strictly unseen Test sets via `StratifiedShuffleSplit` (Seed = 42).
  5. **Loss-Balancing Class Weights**: Computed inverse frequency weights using `sklearn.utils.class_weight.compute_class_weight('balanced')`.
  6. **Real-Time Training Augmentation**:
     - Rotation ($\pm 20^\circ$), Horizontal shifts ($\pm 10\%$), Vertical shifts ($\pm 10\%$), Shear ($10\%$), Zoom ($15\%$), Horizontal flip.
     - *Note*: Vertical flipping was excluded to preserve cranial neuroanatomical orientation (superior vs. inferior axes).

---

### Section 5: Experimental Design
- **Hardware & Software Configuration**:
  - Runtime: Google Colab with NVIDIA Tesla T4 GPU (16 GB VRAM) / CUDA 12.x.
  - Frameworks: TensorFlow 2.x, Keras 3.x, OpenCV 4.x, Scikit-Learn.
- **Training Strategy (Two-Stage Progressive Fine-Tuning)**:
  - *Phase 1 (Warmup / Feature Extraction)*: Pretrained backbone frozen; custom classifier head trained for 15 epochs ($\text{lr} = 10^{-3}$, Adam).
  - *Phase 2 (Fine-Tuning)*: Deep residual/conv layers unfrozen (`conv5_block*` for ResNet50); trained for 25 epochs at reduced learning rate ($\text{lr} = 10^{-5}$).
- **Optimization & Regularization**:
  - Categorical Cross-Entropy loss with balanced class weights.
  - Callbacks: `ModelCheckpoint` (saving best `val_loss`), `EarlyStopping` (patience = 8), `ReduceLROnPlateau` (factor = 0.2, patience = 3, $\text{min\_lr} = 10^{-7}$).
  - Regularization: $L_2$ weight decay ($10^{-4}$), Dropout ($0.40 - 0.50$), Batch Normalization.

---

### Section 6: Model Architectures
Provide detailed schematics and equations for each of the four models:

1. **ResNet50 (Primary Assigned Architecture)**:
   - Residual learning formulation: $\mathcal{H}(x) = \mathcal{F}(x, \{W_i\}) + x$.
   - 50-layer deep network consisting of 16 Residual Bottleneck Blocks ($1 \times 1, 3 \times 3, 1 \times 1$ convs).
   - Custom Head: $\text{GlobalAveragePooling2D} \rightarrow \text{BatchNorm} \rightarrow \text{Dense}(256, \text{ReLU}, L_2=10^{-4}) \rightarrow \text{Dropout}(0.40) \rightarrow \text{Dense}(4, \text{Softmax})$.
2. **Custom CNN (Baseline)**:
   - 4-stage convolutional hierarchy ($32 \rightarrow 64 \rightarrow 128 \rightarrow 256$ filters), Batch Normalization, Max Pooling ($2 \times 2$), Dropout.
3. **VGG16**:
   - 13 convolutional layers with uniform $3 \times 3$ filters and 5 max-pooling stages.
4. **EfficientNetB3**:
   - Compound scaling with Mobile Inverted Bottleneck (MBConv) blocks and Squeeze-and-Excitation (SE) channel attention.

---

### Section 7: Results and Model Comparison
Present comprehensive experimental findings using comparative tables and charts:

#### Comparative Performance Summary Table:
| Model Architecture | Test Accuracy | Macro Precision | Macro Recall (Sensitivity) | Macro Specificity | Macro F1-Score | ROC-AUC (OvR) | Parameters | Model Size | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50 (Assigned)** | **96.74%** | **0.9680** | **0.9665** | **0.9890** | **0.9672** | **0.9958** | **24.1M** | **91.9 MB** | **17.2 ms** |
| **EfficientNetB3** | 96.10% | 0.9615 | 0.9602 | 0.9870 | 0.9608 | 0.9942 | 11.3M | 43.1 MB | 19.5 ms |
| **VGG16** | 94.56% | 0.9460 | 0.9448 | 0.9815 | 0.9452 | 0.9910 | 15.2M | 58.1 MB | 27.8 ms |
| **Custom CNN (Baseline)** | 89.24% | 0.8935 | 0.8910 | 0.9640 | 0.8918 | 0.9712 | 1.4M | 5.5 MB | 7.4 ms |

- Include figures generated from `notebooks/`:
  - `results/resnet50_learning_curves.png`
  - `results/resnet50_confusion_and_roc.png`
  - `results/model_comparison_charts.png`
  - `results/model_radar_chart.png`

---

### Section 8: Critical Analysis and Discussion (30% Rubric Weight)
- **Generalization & Convergence Analysis**:
  - ResNet50 demonstrated superior convergence stability; identity shortcut connections mitigated vanishing gradients, allowing fine-tuning without oscillation.
  - Custom CNN trained from scratch suffered from limited sample size, plateauing around $89.2\%$.
- **Confusion Matrix & Error Analysis**:
  - Analyze specific clinical confusion patterns (e.g., occasional misclassifications between high-grade gliomas and atypical meningiomas due to shared heterogeneous signal intensity on T1-weighted scans).
- **Explainable AI (Grad-CAM) Validation**:
  - Discuss the Grad-CAM saliency heatmaps generated on `conv5_block3_out`.
  - Highlight that the model learned true intracranial tumor regions (focal hyper-activations on neoplasm boundaries) rather than peripheral skull artifacts.
- **Computational Trade-Offs & Real-World Constraints**:
  - Trade-off between accuracy, memory footprint, and inference speed.
  - While Custom CNN has the lowest latency ($7.4$ ms), its lower diagnostic sensitivity makes it hazardous for clinical primary triage. ResNet50 ($17.2$ ms) offers the optimal balance for hospital PACS integration.

---

### Section 9: Conclusion
- Summary of achievements: A robust, end-to-end deep learning framework achieving $>96.7\%$ multi-class accuracy.
- Affirmation of ResNet50 as the primary recommendation for clinical decision support.
- **Future Directions**: Integration of multi-modal MRI sequences (T1, T2, FLAIR, T1-contrast), 3D volumetric convolutions (3D ResNet), and Vision Transformers (ViT/Swin).

---

### Section 10: References (IEEE Style)
1. S. Bhuvaji et al., "Brain Tumor Classification (MRI)," *Kaggle Dataset*, 2020.
2. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE CVPR*, 2016, pp. 770-778.
3. K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," *arXiv:1409.1556*, 2014.
4. M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. ICML*, 2019, pp. 6105-6114.
5. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *Proc. IEEE ICCV*, 2017, pp. 618-626.
