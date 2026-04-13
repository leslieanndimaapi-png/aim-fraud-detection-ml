
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def make_predictions(model, X_test, model_name="Model"):
    """Makes predictions on the test set using the given model."""
    print(f"Making predictions for {model_name}...")
    y_pred = model.predict(X_test)
    print("Predictions complete.")
    return y_pred

def evaluate_classification_report(y_test, y_pred, model_name="Model"):
    """Prints the classification report for model evaluation."""
    print(f"
Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred))

def plot_confusion_matrix(y_test, y_pred, model_name="Model", cmap='Blues'):
    """Plots the confusion matrix for model predictions."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=['Predicted Legitimate', 'Predicted Fraudulent'],
                yticklabels=['Actual Legitimate', 'Actual Fraudulent'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def calculate_comprehensive_metrics(model_name, model, X_test, y_test, is_probability_model=True):
    """Calculates and returns a dictionary of comprehensive evaluation metrics."""
    y_pred = model.predict(X_test)

    if is_probability_model:
        # For models that output probabilities (e.g., RandomForest, XGBoost, LogisticRegression)
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For models that output decision function (e.g., LinearSVC)
        y_proba = model.decision_function(X_test)

    # Calculate metrics for the positive class (fraud=1)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    return {
        'Model': model_name,
        'Precision (Fraud)': f'{precision:.3f}',
        'Recall (Fraud)': f'{recall:.3f}',
        'F1-Score (Fraud)': f'{f1:.3f}',
        'ROC-AUC': f'{roc_auc:.3f}',
        'PR-AUC': f'{pr_auc:.3f}'
    }
