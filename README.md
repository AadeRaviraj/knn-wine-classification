# Wine Quality Classifier — KNN Case Study

![Python](https://img.shields.io/badge/Python-3.x-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Algorithm](https://img.shields.io/badge/Algorithm-KNN-purple)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A complete machine learning project that classifies wine quality based on its chemical properties using the **K-Nearest Neighbors (KNN)** algorithm. The project includes hyperparameter tuning to automatically find the best value of K for optimal accuracy.

---

## Problem Statement

Given a set of chemical properties of wine (such as alcohol content, acidity, pH, etc.), can a machine learning model correctly classify the quality of the wine?

---

## Project Workflow

This project follows a structured 12-step machine learning pipeline:

| Step | Description |
|------|-------------|
| 1 | Load the dataset from a CSV file |
| 2 | Clean the dataset by removing empty/null rows |
| 3 | Separate Independent (X) and Dependent (Y) variables |
| 4 | Split data into Training (80%) and Testing (20%) sets |
| 5 | Apply Feature Scaling using Standard Scaler |
| 6 | Hyperparameter Tuning — test K values from 1 to 20 |
| 7 | Plot K Value vs Accuracy graph |
| 8 | Identify the best value of K |
| 9 | Build the final model using the best K |
| 10 | Calculate final model accuracy |
| 11 | Display Confusion Matrix |
| 12 | Display Classification Report |

---

## Dataset

**File:** `WinePredictor.csv`

**Features (Independent Variables - X):**
All chemical property columns present in the dataset (e.g., alcohol, acidity, pH, sulphates, etc.)

**Target (Dependent Variable - Y):**
- `Class` — the wine quality category

---

## Why Feature Scaling?

KNN is a **distance-based algorithm**. Without scaling, features with larger numerical values will unfairly dominate the distance calculation. `StandardScaler` transforms all features to have:
- Mean = 0
- Standard Deviation = 1

This ensures every feature contributes equally to the prediction.

---

## Hyperparameter Tuning

Instead of manually guessing the value of K, this project tests all K values from **1 to 20**, records the accuracy for each, and automatically selects the K with the highest accuracy.

A line graph is plotted to visually show how accuracy changes with different K values.

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | K-Nearest Neighbors (KNN) |
| K Selection | Automatic (best from 1 to 20) |
| Train/Test Split | 80% / 20% |
| Stratified Split | Yes |
| Feature Scaling | StandardScaler |
| Library | scikit-learn |

---

## Evaluation Metrics

The model is evaluated using:

- **Accuracy Score** — overall percentage of correct predictions
- **Confusion Matrix** — breakdown of correct vs incorrect predictions per class
- **Classification Report** — Precision, Recall, and F1-Score per class

---

## Tech Stack

- Python 3
- pandas — data loading and cleaning
- matplotlib — K vs Accuracy graph
- scikit-learn — model building, scaling, and evaluation

---

## How to Run

1. Clone this repository
2. Place `WinePredictor.csv` in the same folder as the script
3. Install the required libraries:
   ```bash
   pip install pandas matplotlib scikit-learn
   ```
4. Run the script:
   ```bash
   python WineClassifierKNNModelVisualizationFinal.py
   ```

---

## Key Concepts Covered

- Supervised Machine Learning
- Multi-class Classification
- Data Cleaning (handling null values)
- Feature Scaling (StandardScaler)
- Hyperparameter Tuning (finding best K)
- Train/Test Split with Stratification
- K-Nearest Neighbors Algorithm
- Model Evaluation (Accuracy, Confusion Matrix, Classification Report)

 

---

## Author

**Raviraj Aade**

Built as part of a **Machine Learning Case Study** series to understand distance-based classification algorithms and the importance of feature scaling and hyperparameter tuning.
