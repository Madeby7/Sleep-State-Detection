# Sleep State Classification using Late Fusion Deep Learning

This repository contains a complete **late fusion deep learning pipeline** for sleep-wake state classification using multimodal wearable sensor data. The system processes each sensor modality through dedicated sub-networks before fusing features at a higher level for robust sleep state detection.

The project includes **data preprocessing**, **signal filtering**, **late fusion architecture**, and **cross-validation training** using synchronized multimodal sensor streams from wearable devices.

---

## Project Overview

This repository implements a **late fusion sleep state classification system** designed for multimodal wearable devices. The approach processes each sensor modality independently through specialized sub-models before combining features for final classification.

**Key Features:**
- **Late Fusion Architecture**: Separate sub-models for each sensor modality with feature concatenation
- **Sensor-Specific Processing**: Dedicated CNN-BiLSTM architectures tailored to each modality
- **Multimodal Sensor Integration**: LED/PPG, Accelerometer, Gyroscope, Temperature
- **Deep Learning Models**: 
  - PPG Sub-model: Conv2D + BiLSTM (larger architecture)
  - Gyroscope Sub-model: Conv2D + BiLSTM (larger architecture)
  - Accelerometer Sub-model: Conv2D + BiLSTM (larger architecture)
  - Temperature Sub-model: Lightweight Conv2D + Dense layers
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
Late_Fusion/
│
├── Train/
│   ├── Late_Fusion.ipynb          # Main training & evaluation notebook 
│   ├── config.json                # Configuration (sensors, filters, windowing)
│   ├── Data_loader_new.py         # Data loading & preprocessing utilities
│   ├── model.py                   # Late fusion model architecture definitions
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

Centralized configuration for preprocessing and training:

- **Feature definitions**:  
  Accelerometer (3-axis), Gyroscope (3-axis), PPG (IR/Red/Green), Temperature

- **Windowing parameters**:  
  60s windows, 30s step size, batch size 32

- **Sampling frequencies**:  
  Motion sensors @ 25 Hz, Temperature @ 0.2 Hz

- **Filter settings**:  
  Configurable per-sensor (highpass/bandpass, cutoff frequencies)

- **Segmentation**:  
  30s segment filtering for quality control


## 2. Data Processing Pipeline (`Data_loader_new.py`)

Complete preprocessing workflow for **Late Fusion**:


### Signal Filtering

- Butterworth filters (bandpass/highpass) with configurable parameters per modality
- **Accelerometer/Gyroscope**: Highpass 0.1 Hz, order 5
- **PPG**: Bandpass 0.2–5 Hz, order 5



### Artifact Removal & Synchronization

- PPG quality assessment using SNR-based segment filtering
- Percentile-based outlier detection and interpolation (optional)
- Temperature upsampling (0.2 Hz → 25 Hz via linear interpolation)
- Unix timestamp-based sensor alignment with gap detection



### Feature Scaling & Windowing

- RobustScaler for normalization across all sensors
- Sliding window approach with configurable step size
- Segment filtering: Removes windows with \<30s valid data
- Majority voting for binary label assignment per window



### Late Fusion Data Generator

- `process_each_file()`: Generates 4 separate sensor inputs  
  (ACC, GYRO, PPG, TEMP)
- **Output format**:  
  `((acc, gyro, ppg, temp), label)` — tuple of 4 inputs + label


---

## 3. Model Architecture (`model.py`)

**Late Fusion Architecture** via `build_combined_model()`:

Each sensor modality has a dedicated sub-model that processes data independently before fusion:



### PPG Sub-model (`build_sensor_model_2d_2`)

- **Conv2D Block 1**:
  - Conv2D: `num_filters_1` filters, configurable kernel, same padding
  - Batch Normalization
  - LeakyReLU activation

- **Conv2D Block 2**:
  - Conv2D: `num_filters_2` filters, configurable kernel, same padding
  - Batch Normalization
  - LeakyReLU activation
  - MaxPooling2D

- **Conv2D Block 3**:
  - Conv2D: `num_filters_3` filters, configurable kernel, same padding
  - Batch Normalization
  - LeakyReLU activation
  - MaxPooling2D

- **Reshape Layer**: Flattens spatial dimensions for LSTM input

- **Bidirectional LSTM**: `lstm_units` units with return_sequences=True

- **GlobalMaxPooling1D**: Extracts salient temporal features



### Gyroscope Sub-model (`build_sensor_model_2d_2`)

- Same architecture as PPG sub-model
- Tailored for 3-axis rotational motion data (X, Y, Z)
- Independent feature learning for gyroscope-specific patterns



### Accelerometer Sub-model (`build_sensor_model_2d_2`)

- Same architecture as PPG sub-model
- Tailored for 3-axis linear acceleration data (X, Y, Z)
- Independent feature learning for motion-specific patterns



### Temperature Sub-model (`build_sensor_model_2d_sm_2`)

- **Conv2D Block**:
  - Conv2D: `num_filters_1_sm` filters, configurable kernel, same padding
  - Batch Normalization
  - LeakyReLU activation
  - MaxPooling2D

- **Dense Processing**:
  - Flatten layer
  - Dense: 32 units, ReLU activation
  - Reshape for GlobalMaxPooling1D compatibility

- **GlobalMaxPooling1D**: Feature extraction

- Lightweight architecture due to single-channel, low-frequency temperature input (0.2 Hz)



### Feature Fusion Layer

- **Concatenation**: Combines extracted features from all 4 sub-models

- **Dense Block**:
  - Dense: `dense_units` units (configurable), ReLU activation
  - Dropout regularization

- **Output Layer**:
  - Dense: 1 unit, Sigmoid activation
  - Binary classification (Awake=0 / Sleep=1)



### Regularization & Optimization

- **Regularization**:  
  L2 regularization on all Conv2D layers, Dropout after fusion layer

- **Loss Function**:  
  Binary Focal Crossentropy with configurable gamma and alpha parameters

- **Optimizer**:  
  Adam with Cosine Decay learning rate schedule (configurable decay steps and alpha)

- **Metrics**:  
  AUC, Specificity @ 99% Sensitivity, Sensitivity @ 99% Specificity

---

## Key Difference from Early Fusion

**Late Fusion** processes each sensor modality through **separate dedicated sub-models** (4 independent CNN-BiLSTM networks), allowing sensor-specific feature learning tailored to each modality's unique characteristics. Features are concatenated **after** individual processing.

**Early Fusion** concatenates all sensor data **before** the model (at input level), then processes everything through a **single unified CNN-BiLSTM architecture**, treating all modalities as a combined signal from the start.


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

Open and run `Late_Fusion.ipynb` for the complete training and evaluation workflow:

1. **Data Loading**: Configure fold paths and create Late Fusion datasets
2. **Model Building**: Define input shapes for each sensor and build the combined model
3. **Training**: Fit the model with callbacks  
   (checkpoint, early stopping, TensorBoard)
4. **Evaluation**: Generate predictions, calculate metrics, visualize results

---

## Model Naming Convention (Briefly)

```text
fold{N}_LF_G{gamma}a{alpha}_win{W}_s{S}_b{B}_lr{LR}.h5
```

### Where

- **lateFusion**: Late Fusion architecture
- **WithTemp**: Temperature sensor included
- **removedNS**: Noise segments removed via PPG filtering
- **highpass01**: Highpass filter cutoff for ACC / GYRO (0.1 Hz)
- **order5**: Butterworth filter order 5
- **NoPeakRemoval**: Peak removal disabled
- **ppghighpass02low5**: PPG bandpass filter (0.2–5 Hz)
- **RobScaleAll**: RobustScaler normalization applied to all sensors
- **G6a02**: Focal Loss gamma = 6, alpha = 0.2
- **lrdecayCos12ka05**: Cosine decay LR schedule (12k steps, alpha = 0.5)
- **win60**: 60-second windows
- **step30**: 30-second step size
- **batch32**: Batch size 32
- **FilterSegment30**: Minimum segment length = 30 seconds
- **lr0.0001**: Learning rate = 0.0001

---

## Model Performance

### Results (Average 5 Folds)

| Metric                  | Value  |
|-------------------------|--------|
| ROC AUC                 | 0.9094 |
| F1 Score                | 0.9075 |
| Accuracy                | 85.63% |
| Sensitivity (Recall)    | 85.65% |
| Specificity             | 85.52% |


### Class-Level Performance

| Class | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| Awake | 0.52 | 0.86 | 0.64 | 1844 |
| Sleep | 0.97 | 0.86 | 0.91 | 10308 |

---
