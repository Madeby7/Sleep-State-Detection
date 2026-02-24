# Sleep State Classification using Early Fusion Deep Learning

This repository contains a complete **early fusion deep learning pipeline** for sleep-wake state classification using multimodal wearable sensor data. The system integrates physiological and motion sensor signals through a unified CNN-BiLSTM architecture for robust sleep state detection.

The project includes **data preprocessing**, **signal filtering**, **early fusion architecture**, and **cross-validation training** using synchronized multimodal sensor streams from wearable devices.

---

## Project Overview

This repository implements an **early fusion sleep state classification system** designed for multimodal wearable devices. The approach combines multiple sensor modalities at the input level before feeding into a unified deep learning model.

**Key Features:**
- **Early Fusion Architecture**: Concatenates all sensor modalities before model processing
- **Multimodal Sensor Integration**: LED/PPG, Accelerometer, Gyroscope, Temperature
- **Deep Learning Model**: CNN layers + Bidirectional LSTM + Attention mechanisms
- **Cross-Validation Framework**: 5-fold cross-validation with ADHD dataset
- **Advanced Signal Processing**: Butterworth filtering, peak removal, artifact detection
- **Class Imbalance Handling**: Binary Focal Loss with configurable gamma and alpha
- **Learning Rate Scheduling**: Cosine decay with warm restarts

### Supported Sensors:
- **LED/PPG signals** (IR, Red, Green channels) @ 25 Hz
- **Accelerometer** (X, Y, Z axes) @ 25 Hz
- **Gyroscope** (X, Y, Z axes) @ 25 Hz
- **Temperature** (Object temperature) @ 0.2 Hz

---

## Environment Setup

### Python Version
- **Python 3.11** (recommended)


### GPU / CUDA Requirements

This project was tested with the following GPU stack:

- NVIDIA Driver: supports CUDA 11.X
- CUDA Toolkit (nvcc): 11.7
- PyTorch: 2.9.0
  - CUDA runtime: 12.8
  - cuDNN: 9.10.2 (enabled)
- TensorFlow: 2.10.0
  - Official CUDA support: 11.2
  - cuDNN: 8.1
  - Note: TensorFlow may run in CPU-only mode in this environment.

### Verify CUDA & cuDNN

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.version.cuda, torch.backends.cudnn.version())"
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"

```
---

## Repository Structure
```text
Early_Fusion/
│
├── Train/
│   ├── Early_Fusion.ipynb         # Main training & evaluation notebook
│   ├── config.json                # Configuration (sensors, filters, windowing)
│   ├── Data_loader_new.py         # Data loading & preprocessing utilities
│   ├── model.py                   # Early fusion model architecture definitions
│   └── ppg_preprocess.py          # PPG-specific preprocessing functions
│
├── data/                          # Dataset directory (private)
│   └── 5folds_adhd/               # 5-fold cross-validation splits
│       ├── fold1/
│       ├── fold2/
│       ├── fold3/
│       ├── fold4/
│       └── fold5/
│
├── models/                        # Saved trained models
│   ├── fold_1/
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   └── fold_5/
│
├── results/                       # Evaluation results
│   ├── fold_1/
│   │   ├── *_classification_report.txt
│   │   ├── *_Conf_mat.png
│   │   ├── *_roc_curve.png
│   │   └── *_predictions.csv
│   ├── fold_2/
│   ├── fold_3/
│   ├── fold_4/
│   └── fold_5/
│
└── README.md

```
### Note:
The dataset used in this project is private and is not included in this repository.
To request access to the data, please contact the author.

---

## Core Functionalities

### 1. Configuration Management (`config.json`)

Centralized configuration for preprocessing, feature fusion, and training
in the **Early Fusion** pipeline:

- **Feature definitions**:  
  Accelerometer (3-axis), Gyroscope (3-axis), PPG (IR / Red / Green), Temperature

- **Windowing parameters**:  
  60s windows, 30s step size, batch size 32

- **Sampling frequencies**:  
  Motion sensors @ 25 Hz, Temperature @ 0.2 Hz

- **Filter settings**:  
  Configurable per sensor  
  (highpass / bandpass filters with adjustable cutoff frequencies)

- **Segmentation**:  
  30s segment filtering for signal quality control

- **Early Fusion strategy**:  
  All sensor modalities are temporally aligned, normalized, and **concatenated
  at the feature level** to form a single fused input tensor before model training

---
## 2. Data Processing Pipeline (`Data_loader_new.py`)

Complete preprocessing workflow for **Early Fusion**:



### Signal Filtering

- Butterworth filters (bandpass / highpass) with configurable parameters per modality
- **Accelerometer / Gyroscope**: Highpass 0.1 Hz, order 5
- **PPG**: Bandpass 0.2–5 Hz, order 5



### Artifact Removal & Synchronization

- PPG quality assessment using SNR-based segment filtering
- Percentile-based outlier detection and interpolation (optional)
- Temperature upsampling (0.2 Hz → 25 Hz via linear interpolation)
- Unix timestamp-based sensor alignment with gap detection



### Feature Scaling & Windowing

- StandardScaler for normalization across all sensors
- Sliding window approach with configurable step size
- Segment filtering: Removes windows with \<30s valid data
- Majority voting for binary label assignment per window

---

## 3. Model Architecture (`model.py`)

**Early Fusion Architecture** via `build_stacked_model()`:

All sensor modalities are concatenated at the input level and processed through a unified architecture:



### Unified Model Architecture

- **Input Layer**:  
  Concatenated tensor from all modalities (ACC + GYRO + PPG + TEMP)  
  Shape: `(1500, 10, 1)` — 1500 time steps × 10 channels × 1

- **Conv2D Block 1**:  
  - Conv2D: `num_filters_1` filters, kernel (3×3), same padding
  - Batch Normalization
  - LeakyReLU activation

- **Conv2D Block 2**:  
  - Conv2D: `num_filters_2` filters, kernel (3×3), same padding
  - Batch Normalization
  - LeakyReLU activation
  - MaxPooling2D: (2×2)
  - Dropout

- **Conv2D Block 3**:  
  - Conv2D: `num_filters_3` filters, kernel (3×3), same padding
  - Batch Normalization
  - LeakyReLU activation
  - MaxPooling2D: (2×1)
  - Dropout

- **Reshape Layer**:  
  Flattens spatial dimensions: `(time_steps, features × channels)`

- **Bidirectional LSTM**:  
  - `lstm_units` units (configurable, typically 128)
  - return_sequences=True for temporal modeling
  - GlobalMaxPooling1D for feature extraction

- **Dense Block**:  
  - Dense: `dense_units` units (configurable), ReLU activation
  - Dropout regularization

- **Output Layer**:  
  - Dense: 1 unit, Sigmoid activation
  - Binary classification (Awake=0 / Sleep=1)


### Regularization & Optimization

- **Regularization**:  
  L2 regularization on Conv2D layers, Dropout after pooling and dense layers

- **Loss Function**:  
  Binary Focal Crossentropy (gamma, alpha configurable)

- **Optimizer**:  
  Adam with Cosine Decay learning rate schedule

- **Metrics**:  
  AUC, Specificity @ 99% Sensitivity, Sensitivity @ 99% Specificity

---

## Key Difference from Late Fusion

**Early Fusion** concatenates all sensor data **before** the model (at input level), then processes everything through a **single unified CNN-BiLSTM architecture**. 

**Late Fusion** processes each sensor modality through **separate dedicated sub-models** (4 independent CNN-BiLSTM networks), then concatenates the extracted features before the final classification layers. This allows sensor-specific feature learning tailored to each modality's characteristics.


---

## Fixed Configuration Parameters

Based on extensive hyperparameter tuning, the following parameters are provided optimal performance. (For detailed hyperparameter experiment you are free to change)

```python
FIXED_PARAMS = {
    'window_size_seconds': 60,
    'step_size_seconds': 30,      # Training: 30s, Testing: 60s
    'segment_length_seconds': 30, # Min valid segment length
    'batch_size': 32,
    'learning_rate': 0.0001,
    'focal_loss_gamma': 6,
    'focal_loss_alpha': 0.2,
}

```
---

## Usage

### Training & Evaluation

Open and run `Early_Fusion.ipynb` for the complete training and evaluation workflow:

1. **Data Loading**: Configure fold paths and create Early Fusion datasets
2. **Model Building**: Define input shape for concatenated modalities and build the unified model
3. **Training**: Fit the model with callbacks  
   (checkpoint, early stopping, TensorBoard)
4. **Evaluation**: Generate predictions, calculate metrics, visualize results


---


## Model Naming Convention (Briefly)

```text
fold{N}_modelLstm{L}_earlyFusion_WithTemp_removedNS_highpass{H}_order{O}_NoPeakRemoval_ppghighpass{PH}low{PL}_StanScaleAll_BinFocalLossG{gamma}a{alpha}_lrdecayCos{D}ka{A}_win{W}_step{S}_batch{B}_FilterSegment{F}_lr{LR}.h5
```

### Where

- **EF**: Early Fusion
- **Lstm128**: BiLSTM with 128 units
- **WithTemp**: Temperature sensor included
- **removedNS**: Noise segments removed
- **highpass01**: Highpass filter at 0.1 Hz (ACC / GYRO)
- **order5**: Butterworth filter order 5
- **ppghighpass02low5**: PPG bandpass 0.2–5 Hz
- **StanScaleAll**: StandardScaler normalization
- **G6a02**: Focal Loss gamma = 6, alpha = 0.2
- **lrdecayCos12ka01**: Cosine decay, 12k steps, alpha = 0.1
- **win60**: 60-second windows
- **step30**: 30-second step size
- **batch32**: Batch size 32
- **FilterSegment30**: Minimum 30s valid segment
- **lr0.0001**: Learning rate 0.0001


---

## Model Performance

### Results (Average 5 folds)

| Metric               | Value  |
|----------------------|--------|
| ROC AUC              | 0.8735 |
| F1 Score             | 0.9033 |
| Accuracy             | 84.77% |
| Sensitivity (Recall) | 85.97% |
| Specificity          | 77.66% |



### Class-Level Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Awake | 0.48 | 0.77 | 0.59 | 1746 |
| Sleep | 0.96 | 0.86 | 0.91 | 10353 |

---



