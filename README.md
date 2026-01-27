# ECG Signal Classification using Wavelet Transform & XGBoost

## Project Overview
This project implements a Machine Learning pipeline to classify Electrocardiogram (ECG) signals into three distinct cardiac conditions:
1.  **Arrhythmia (ARR)**
2.  **Congestive Heart Failure (CHF)**
3.  **Normal Sinus Rhythm (NSR)**

The solution utilizes **Discrete Wavelet Transform (DWT)** for advanced feature extraction and **XGBoost** for classification, achieving an overall accuracy of **90%**.

## Key Features
* **Wavelet-Based Analysis:** Uses `sym5` wavelet for both signal denoising (Soft Thresholding) and multi-level feature extraction.
* **No Data Leakage:** Implements **Patient-Level Splitting** (splitting by record ID before segmentation) to ensure the model is tested on unseen patients, ensuring medical validity.
* **Imbalance Handling:** Utilizes **ADASYN** (Adaptive Synthetic Sampling) to address class imbalance, significantly improving detection of the minority class (CHF).
* **High Performance:** Tuned XGBoost model (`max_depth=10`, `n_estimators=500`) achieves superior results compared to baseline models.

## Dataset
The project uses the PhysioNet ECG dataset.
* **Sampling Rate:** 128 Hz
* **Format:** Raw CSV values (`ECGData.csv`) and Labels (`ECGDataLabel.csv`).
* **Input Shape:** 162 records with variable lengths (segmented into 10-second windows during processing).

## Methodology Pipeline
1.  **Preprocessing (Denoising):**
    * Applied **Wavelet Soft Thresholding** using the `sym5` wavelet (Level 2) to remove high-frequency noise and baseline wander.
2.  **Feature Extraction:**
    * Decomposed signals using DWT (`sym5`, Level 5).
    * Extracted **10 features per sub-band** (Approximation & Details):
        * *Statistical:* Mean, Std Dev, Max, Min, RMS, Skewness, Kurtosis.
        * *Signal Characteristics:* Shannon Entropy, Relative Energy, Zero Crossing Rate (ZCR).
3.  **Segmentation:**
    * Signals are sliced into **10-second non-overlapping segments**.
4.  **Class Balancing:**
    * Applied **ADASYN** on the training set to generate synthetic samples for the minority classes (CHF and NSR).
5.  **Classification:**
    * Model: **XGBoost Classifier**.
    * Hyperparameters: `n_estimators=500`, `max_depth=10`, `learning_rate=0.05`.

## Results
The model was evaluated on a strictly held-out test set (stratified by patient).
| Metric | Value |
| :--- | :--- |
| **Accuracy** | **90.00%** |
| **F1-Score (NSR)** | **0.96** |
| **F1-Score (ARR)** | **0.92** |
| **F1-Score (CHF)** | **0.74** |

## Installation & Usage
### 1. Clone the Repository
```bash
git clone https://github.com/fzhnd/wavelet-ecg-classification.git
cd wavelet-ecg-classification
```
### 2. Install Dependencies
```bash
pip install numpy pandas scipy scikit-learn xgboost imbalanced-learn PyWavelets matplotlib seaborn
```
### 3. File Structure
Ensure directory looks like this:
```
├── data/
│   ├── ECGData.csv
│   └── ECGDataLabel.csv
├── confusion-matrix.png
├── wavelet-ecg.ipynb
└── README.md
```
### 4. Run the Project
Open the notebook and run all cells:
```bash
jupyter notebook wavelet-ecg.ipynb

```
