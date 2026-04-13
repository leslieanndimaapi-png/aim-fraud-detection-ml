# AIM Capstone Project: Fraud Detection ML

## 📌 Project Overview
A financial services company that issues credit cards processes millions of transactions daily. With the growth of digital payments, fraudulent transactions have also increased and attacks are continuously evolving that leads to financial losses and reduced customer trust. It is critical that credit card companies are able to detect fraudulent transactions so that customers are not charged for items that they did not purchase.

The goal of this capstone project is to develop a machine learning model that can predict whether a credit card transaction is fraudulent or legitimate by exploring different supervised (classification) and unsupervised (anomaly detection) techniques.

To evaluate the performance of the fraud detection model, we will be using the following key metrics:

Precision and recall to balance accurate fraud detection with false alarms
f1 Score for imbalanced dataset evaluation
ROC-AUC which measures the model’s ability to distinguish between classes across various thresholds

The primary objective is to **maximize detection of fraudulent transactions (high recall)** while maintaining an acceptable level of precision.

---

## 📊 Dataset
- Source: Kaggle – Fraud Detection Dataset (kartik2112/fraud-detection)
- Contains transaction-level data with labeled fraudulent and non-fraudulent instances
- Highly imbalanced dataset (fraud cases are rare)

---

## ⚙️ Project Pipeline

### 1. Data Preprocessing
- Handled missing values, duplicates, and outliers
- Performed train-test split
- Addressed class imbalance using:
  - Class weighting
  - SMOTE (Synthetic Minority Oversampling Technique)

### 2. Exploratory Data Analysis (EDA)
- Analyzed feature distributions
- Compared fraud vs non-fraud transactions
- Identified key patterns and correlations

### 3. Feature Engineering
- Scaling and encoding of features
- Creation of domain-based features
- Feature selection to retain most relevant variables

### 4. Model Implementation
Models explored:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LinearSVC

Evaluation metrics:
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC (primary metric due to class imbalance)

### 5. Explainability & Feature Importance
- SHAP (SHapley Additive exPlanations) used for:
  - Global feature importance
  - Local explanation of predictions

### 6. Bias & Fairness Analysis
- Evaluated potential bias across available features
- Discussed fairness metrics and limitations

---

## 🎯 Key Results
- SMOTE improved recall, allowing better detection of fraudulent transactions
- Class weighting maintained higher precision
- Final model selected based on recall-focused performance and PR-AUC

---

## ⚠️ Limitations
- Severe class imbalance may still affect generalization
- SMOTE introduces synthetic data which may not fully reflect real fraud patterns
- Model performance may degrade over time as fraud patterns evolve

---

## 📁 Repository Structure
```bash
aim-fraud-detection-ml/
├── data/                       # Raw and processed data
├── notebooks/                  # pre-prcessing, EDA, feature engineering, modeling, explainability
├── src/                        # Modular scripts for pipeline
├── models/                     # Saved trained models - configs and parameters
├── reports/                    # Project reports
├── requirements.txt
├── README.md
└── .gitignore
```