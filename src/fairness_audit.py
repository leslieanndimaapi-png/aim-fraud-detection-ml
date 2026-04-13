
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def calculate_fairness_metrics(data, group_col, target_col='is_fraud', prediction_col='y_pred'):
    """Calculates fairness metrics (Recall, Precision, FNR, FPR) for different groups."""
    results = []
    for group_name, group_data in data.groupby(group_col):
        y_true = group_data[target_col]
        y_pred = group_data[prediction_col]

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0) # Handle cases where one class is missing

        # Avoid division by zero
        recall_fraud = tp / (tp + fn) if (tp + fn) != 0 else np.nan
        precision_fraud = tp / (tp + fp) if (tp + fp) != 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) != 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) != 0 else np.nan

        overall_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else np.nan

        results.append({
            group_col: group_name,
            'Count': len(group_data),
            'Fraud Cases': group_data[target_col].sum(),
            'Legit Cases': (len(group_data) - group_data[target_col].sum()),
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Recall (Fraud)': f'{recall_fraud:.3f}',
            'Precision (Fraud)': f'{precision_fraud:.3f}',
            'FNR': f'{fnr:.3f}',
            'FPR': f'{fpr:.3f}',
            'Overall Accuracy': f'{overall_accuracy:.3f}'
        })
    return pd.DataFrame(results)

def create_fairness_dataframe(original_df, y_test, y_pred, age_bins, age_labels):
    """Creates a DataFrame suitable for fairness auditing by merging predictions and sensitive attributes."""
    fairness_df = original_df.loc[y_test.index].copy()
    fairness_df['y_test'] = y_test
    fairness_df['y_pred'] = y_pred

    # Re-calculate age_bin for fairness_df based on original dob and trans_date_trans_time
    fairness_df['trans_date_trans_time'] = pd.to_datetime(fairness_df['trans_date_trans_time'])
    fairness_df['dob'] = pd.to_datetime(fairness_df['dob'])
    fairness_df['age'] = (fairness_df['trans_date_trans_time'].dt.year - fairness_df['dob'].dt.year)
    fairness_df['age_bin'] = pd.cut(fairness_df['age'], bins=age_bins, labels=age_labels, right=False)

    return fairness_df
