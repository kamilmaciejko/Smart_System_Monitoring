# Smart System Monitoring: Real-Time Detection of Workload Modes in Computer Systems Using Machine Learning Methods

## About the project
This repository contains a **data science** and **IoT-related** project focused on developing a predictive model that classifies the workload mode of a computer system based on **real-time monitoring of hardware performance metrics**.

The project serves as a **proof of concept** demonstrating how machine learning methods can be applied to analyse system behaviour and support intelligent resource management in IoT-related environments.

## Key Features

- Real-time system performance monitoring (CPU, GPU, RAM, disk, network)
- Machine learning workload classification
- Comparison of multiple ML models (Decision Trees, ANN, XGBoost)
- Feature engineering with rolling statistics
- Real-time prediction simulation with live visualization
- Proof-of-concept application for IoT resource management

## Project's Summary
*Note: The following text is adapted from the full documentation paper.*

This paper investigates the use of machine learning methods to monitor and classify laptop performance modes in real time. By analysing hardware performance metrics such as CPU and GPU utilisation, a data-driven system was developed to categorize workloads into three operational states: low, standard, and high performance. Several machine learning methods, including decision tree–based models and artificial neural networks, were evaluated, with the XGBoost model achieving the highest classification accuracy.

Beyond model development, the study also includes a practical simulation demonstrating how such predictions can support dynamic adjustment of system performance settings during active device usage. Although the results are based on data collected from a specific hardware configuration, they provide a strong proof of concept for automated and intelligent resource management in IoT-related environments.

## Repository structure
This repo consists of 4 files:

- **Smart System Monitoring Using ML Methods.pdf** - PDF with full documentation of the project, conducted steps of data preparation, modelling and real-time application with results and discussion.
- **final_model.pkl** - Trained XGBoost model wrapped in a GridSearchCV object. Contains the best estimator selected during hyperparameter tuning.
- **smart_system_monitoring_code.ipynb** - Notebook with full code containing functions for data collection, data preprocessing, modelling, evaluation and function for real-time application.
- **training_dataset.csv** - Dataset used for training the models.

## Model

The final model (`final_model.pkl`) is a tuned XGBoost classifier obtained using GridSearchCV.  
It can be loaded with:

```python
import joblib
final_model = joblib.load("final_model.pkl")
```

Below is a simple example showing how to perform a prediction on sample data using the loaded model and training dataset. 

```python
import pandas as pd

# Load example dataset
df = pd.read_csv("training_dataset.csv")

# Separate features and target
X = df.drop(columns=["label"])

# Take a sample observation
sample = X.iloc[[0]]

# Make prediction
prediction = final_model.predict(sample)

print("Predicted workload mode:", prediction[0])
```

### **Important: The model expects input data with the same feature structure used during training *(see: Model Input Features)***

## Model Input Features

The final model was trained using **43 numerical input features** derived from raw system metrics and rolling statistical transformations. The trained model expects the following system performance metrics as input features: 

### CPU Metrics
- **cpu_core_{1–16}_usage** – CPU utilization (%) for each individual core.
- **cpu_usage_avg** – Average CPU utilization across all cores.
- **cpu_speed_mhz** – Current CPU clock speed (MHz).

### GPU Metrics
- **gpu_usage** – GPU utilization (%).
- **gpu_temperature** – GPU temperature (°C).

### Memory Metrics
- **ram_usage** – RAM utilization (%).

### Disk Activity
- **disk_read_bytes_delta** – Change in disk read bytes between measurements.
- **disk_write_bytes_delta** – Change in disk write bytes between measurements.

### Network Activity
- **net_bytes_sent_delta** – Change in bytes sent over the network.
- **net_bytes_recv_delta** – Change in bytes received over the network.

### Rolling Statistics (10-second window)
Rolling mean and standard deviation calculated over a 10-second window to smooth short-term fluctuations.

- **{metric}_10s_mean** – Rolling mean of the metric.
- **{metric}_10s_std** – Rolling standard deviation of the metric.

Applied to:
- `cpu_speed_mhz`
- `gpu_usage`
- `gpu_temperature`
- `ram_usage`
- `cpu_usage_avg`
- `disk_read_bytes_delta`
- `disk_write_bytes_delta`
- `net_bytes_sent_delta`
- `net_bytes_recv_delta`

### Target Variable
- **label** – Workload mode classification (`blue` = low workload, `white` = standard workload, `red` = high workload).

## Credits
Author: Kamil Maciejko <br>
8.03.2026 <br>
Warsaw, SGH
