
import kagglehub
import pandas as pd
import os

def load_fraud_data(dataset_name='kartik2112/fraud-detection', file_name='fraudTrain.csv'):
    """Downloads the fraud detection dataset from Kaggle and loads it into a pandas DataFrame."""
    print(f"Downloading dataset: {dataset_name}")
    path = kagglehub.dataset_download(dataset_name)
    data_file = os.path.join(path, file_name)
    df = pd.read_csv(data_file)
    print("DataFrame loaded successfully.")
    return df

