# 🛠️ Anomaly Detection in Hydraulic Systems using Two-Stage Autoencoders

## 📌 Abstract  
This project presents a robust anomaly detection system for **hydraulic systems** using a **two-stage autoencoder** architecture. It leverages **time-series multi-sensor data** to detect early signs of failure, enabling predictive maintenance and minimizing downtime.

The pipeline involves:  
- **Stage 1**: Learning normal system behavior and reconstructing sensor signals  
- **Stage 2**: Learning from residual errors of Stage 1 to capture subtle and complex anomalies  

This dual-layered design improves detection accuracy on real-world, noisy, and heterogeneous sensor data.

---

## 🔍 Table of Contents
- 🚀 Introduction
- ❗ Problem Statement
- 🎯 Project Objectives
- 📚 Literature Review
- 🧪 Methodology
- 📊 Results
- 🔍 Key Observationds
- ⚙️ Usage

---

## 🚀 Introduction
Hydraulic systems play a vital role in industries like **aerospace**, **automotive**, and **manufacturing**. Failures in these systems can cause costly downtimes. With advances in sensor technology, machine learning models—especially autoencoders—can now be used to detect anomalies early and accurately.

---

## ❗ Problem Statement
- Complex, **high-dimensional**, **multivariate** time-series data  
- **Varying sampling rates** and sensor data shapes  
- Difficulty in modeling and distinguishing **normal vs abnormal** operations  
- Traditional methods fail to generalize or scale to real-world systems

---

## 🎯 Project Objectives
- ✅ Preprocess multi-sensor time-series data  
- ✅ Build a **Two-Stage Autoencoder**:  
  - **Stage 1**: Learn and reconstruct normal behavior  
  - **Stage 2**: Analyze Stage 1 residuals to detect subtle anomalies  
- ✅ Compute and evaluate reconstruction errors for anomaly detection  
- ✅ Visualize anomalies and interpret failure patterns over time

---

## 📚 Literature Review

### 🔎 What is Anomaly Detection?
Anomaly detection refers to identifying data points that **deviate from expected behavior**, such as:
- **Point anomalies**
- **Contextual anomalies**
- **Collective anomalies**

### 🧠 Traditional Techniques
- **Statistical**: Z-score, ARIMA  
- **Distance-based**: k-NN, LOF  
- **Model-based**: HMM, Gaussian Mixtures  
- **Decomposition**: STL, seasonal-trend decomposition  

### ⚙️ Why Two-Stage Autoencoders?
- **Stage 1**: Learns baseline behavior  
- **Stage 2**: Learns from Stage 1 residuals  
- Improves detection of **non-obvious, nuanced anomalies** often missed by traditional or single-stage models

---

## 🧪 Methodology

### 1️⃣ Two-Stage Autoencoder Architecture

**Stage 1:**  
- Learns baseline patterns of normal sensor behavior  
- Trained using **Mean Squared Error (MSE)** loss  
- Output: Reconstructed signal and residuals (difference between original and reconstruction)

**Stage 2:**  
- Trained to model **patterns within the reconstruction errors**  
- Detects **subtle, second-order deviations** that Stage 1 may miss

---

### 🔄 Data Preprocessing

- Load multi-sensor raw time-series dataset  
- Handle missing values and inconsistent sampling rates (if any)  
- Normalize features using `StandardScaler`  
- Visualize data to observe trends, spikes, and noise  
- Split into:
  - **Training Set**: Only normal behavior
  - **Testing Set**: Contains both normal and anomalous behavior

---

### 🧠 Model Training

#### ✅ Stage 1 Autoencoder
- Input: Normalized time-series sensor data  
- Loss: **MSE**  
- Output: Reconstructed signal + Residuals  
- Goal: Learn "normal" operational patterns

#### ✅ Stage 2 Autoencoder
- Input: Residuals from Stage 1  
- Loss: **MSE**  
- Output: Refined reconstruction of residuals  
- Goal: Detect more **complex or subtle anomalies**

---

### 📈 Anomaly Detection

#### 🔹 Threshold Calculation
Use statistical heuristics based on residual error distribution:  
```python
threshold = mean(residual_error) + 3 * std(residual_error)
```

### 🔹 Final Detection Logic

- Compute reconstruction errors from **both Stage 1 and Stage 2**
- Combine the errors (e.g., using a **weighted average** or **sum**)
- Flag any time point as an **anomaly** if the combined error exceeds the threshold
- This **dual-layer architecture** effectively captures both:
  - Obvious, large-scale faults
  - Subtle, complex anomalies that single-stage models may miss

---

## 📊 Results

### ✔️ Evaluation Metrics

- **Mean Squared Error (MSE)** on reconstruction
- **Precision / Recall / F1-Score** (if ground-truth labels are available)
- **Confusion Matrix**: Includes True Positives, False Positives, True Negatives, and False Negatives
- **Anomaly Score Curves**: Line plots to visualize error progression over time

---

### 🔍 Key Observations

- The **two-stage autoencoder** significantly outperforms traditional and single-stage models by:
  - Accurately capturing **complex anomaly patterns**
  - **Reducing false positives** and increasing model robustness
  - Detecting **subtle, non-obvious deviations** in system behavior
- **Visual diagnostics** help identify *when* and *where* anomalies occur, aiding faster root-cause analysis

---

## ⚙️ Getting Started

### 🖥️ Prerequisites

Make sure the following packages are installed in your Python environment:

```bash
Python >= 3.7
TensorFlow or Keras
NumPy
Pandas
Matplotlib
Scikit-learn


