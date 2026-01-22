# Wavelet-Based ECG Classification
This repository implements a **wavelet-based feature extraction and machine learning pipeline** for **ECG signal classification** using data from **PhysioNet**. The goal is to classify ECG recordings into three clinically relevant categories: **Arrhythmia (ARR)**, **Congestive Heart Failure (CHF)**, and **Normal Sinus Rhythm (NSR)**.

The implementation follows practices in biomedical signal processing, including record-level evaluation to avoid data leakage and time–frequency analysis using Discrete Wavelet Transform (DWT).

## Dataset Description
The dataset consists of **162 ECG recordings** with the following characteristics:
* **Sampling frequency**: 128 Hz
* **Recording duration**: 512 seconds per record
* **Total samples per record**: 65,536

### Class Distribution
| Class                    | Label | Number of Records |
| ------------------------ | ----- | ----------------- |
| Arrhythmia               | ARR   | 90                |
| Congestive Heart Failure | CHF   | 30                |
| Normal Sinus Rhythm      | NSR   | 36                |

The ECG signals and labels are provided as:
* `ECGData.csv` – ECG signal recordings
* `ECGDataLabel.csv` – Corresponding class labels

## Methodology Overview
The proposed pipeline consists of the following stages:

1. **Record-Level Train–Test Split**
   Each ECG record is assigned exclusively to either the training or testing set to prevent subject-level information leakage.

2. **ECG Segmentation**
   Each ECG record is segmented into **8-second segments** (1024 samples per segment) to capture local temporal dynamics.

3. **Wavelet-Based Feature Extraction**
   * Discrete Wavelet Transform (DWT)
   * Wavelet: `db4`
   * Decomposition level: 4

4. **Feature Engineering**
   For each wavelet sub-band, the following features are extracted:
   * Mean
   * Standard deviation
   * Skewness
   * Kurtosis
   * Shannon entropy
   * Relative wavelet energy

5. **Classification**
   * Classifier: Support Vector Machine (SVM) with RBF kernel
   * Cost-sensitive learning using `class_weight='balanced'`
   * Hyperparameter tuning via GridSearchCV (macro F1-score)

6. **Evaluation**
   * Accuracy
   * Precision, Recall, F1-score per class
   * Confusion Matrix visualization

## Results
Typical performance obtained using this pipeline:
* **Accuracy**: ~88%
* **Macro F1-score**: ~0.87
* **Recall (CHF)**: ~0.78

These results are competitive with existing wavelet-based ECG classification methods reported in the literature and are obtained without deep learning models.

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/fzhnd/wavelet-ecg-classification.git
   cd wavelet-ecg-classification
   ```

2. **Prepare the files**
   The repository contains the following files:
   ```
   ├── data.zip                  # ECG dataset (compressed)
   ├── ECG-classification.ipynb  # Main Jupyter Notebook
   ├── README.md                 # Project documentation
   ```

   Extract the dataset first:
   ```bash
   unzip data.zip
   ```

   After extraction, make sure the directory contains:
   ```
   ├── ECGData.csv
   ├── ECGDataLabel.csv
   ├── ECG-classification.ipynb
   ├── README.md
   ```

3. **Install required dependencies**
   ```bash
   pip install numpy pandas scipy pywavelets scikit-learn matplotlib seaborn
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook ECG-classification.ipynb
   ```

All preprocessing, feature extraction, model training, and evaluation steps are executed sequentially inside the notebook.

## Visualization
The project includes visualization of the **confusion matrix** using a heatmap to clearly illustrate class-wise classification performance.
