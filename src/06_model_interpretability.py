
import shap
import numpy as np
import pandas as pd

def initialize_shap_explainer(model):
    """Initializes a SHAP TreeExplainer for the given model."""
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    return explainer

def sample_data_for_shap(X_data, sample_size=5000):
    """Samples a subset of the data for SHAP value calculation for computational efficiency."""
    sample_size_actual = min(sample_size, X_data.shape[0])
    shap_sample_indices = np.random.choice(X_data.index, sample_size_actual, replace=False)
    X_data_sample_shap = X_data.loc[shap_sample_indices]
    print(f"Created SHAP sample of size {sample_size_actual}.")
    return X_data_sample_shap

def calculate_shap_values(explainer, X_data_sample_shap, model_name="Model"):
    """Calculates SHAP values for the sampled data using the provided explainer."""
    print(f"Calculating SHAP values for {model_name}...")
    shap_values = explainer.shap_values(X_data_sample_shap)
    print("SHAP values calculation complete.")
    return shap_values
