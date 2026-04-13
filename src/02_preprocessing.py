
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder, StandardScaler

def check_duplicates(df):
    """Checks for and removes duplicate rows in the DataFrame."""
    print("Checking for duplicate rows...")
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        print(f"Found {duplicate_rows} duplicate rows. Removing them...")
        df.drop_duplicates(inplace=True)
        print("Duplicate rows removed.")
    else:
        print("No duplicate rows found.")
    print("Current DataFrame shape after duplicate check:", df.shape)
    return df

def handle_outliers_winsorize(df, column='amt', limits=[0.01, 0.01]):
    """Applies Winsorization to a specified column to handle outliers."""
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame. Skipping Winsorization.")
        return df
    print(f"Original min '{column}': {df[column].min():.2f}")
    print(f"Original max '{column}': {df[column].max():.2f}")
    df[f'{column}_winsorized'] = winsorize(df[column], limits=limits)
    print(f"Winsorized min '{column}': {df[f'{column}_winsorized'].min():.2f}")
    print(f"Winsorized max '{column}': {df[f'{column}_winsorized'].max():.2f}")
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the Haversine distance between two sets of coordinates in kilometers."""
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def engineer_features(df):
    """Engineers date/time, age, and location-based features."""
    df_fe = df.copy()

    # Drop identifier columns that are not useful for modeling
    df_fe = df_fe.drop(columns=['Unnamed: 0', 'cc_num', 'trans_num', 'first', 'last', 'street'], errors='ignore')

    # Convert date columns to datetime objects
    df_fe['trans_date_trans_time'] = pd.to_datetime(df_fe['trans_date_trans_time'])
    df_fe['dob'] = pd.to_datetime(df_fe['dob'])

    # Extract date/time features
    df_fe['trans_hour'] = df_fe['trans_date_trans_time'].dt.hour
    df_fe['trans_day_of_week'] = df_fe['trans_date_trans_time'].dt.dayofweek
    df_fe['trans_month'] = df_fe['trans_date_trans_time'].dt.month
    df_fe['trans_day'] = df_fe['trans_date_trans_time'].dt.day
    df_fe['trans_year'] = df_fe['trans_date_trans_time'].dt.year

    # Calculate age of the cardholder at the time of transaction
    df_fe['age'] = (df_fe['trans_date_trans_time'].dt.year - df_fe['dob'].dt.year)

    # Calculate geographical distance between customer and merchant
    df_fe['distance_to_merchant'] = df_fe.apply(lambda row: haversine_distance(row['lat'], row['long'], row['merch_lat'], row['merch_long']), axis=1)

    # Drop original date/time and geographic coordinate columns after feature extraction
    df_fe = df_fe.drop(columns=['trans_date_trans_time', 'dob', 'unix_time', 'lat', 'long', 'merch_lat', 'merch_long'], errors='ignore')

    print("DataFrame after engineering date/time, age, and distance features:")
    print(df_fe.head())
    return df_fe

def encode_categorical_features(df):
    """Applies Label Encoding to object type columns."""
    categorical_cols_to_encode = df.select_dtypes(include='object').columns.tolist()
    print(f"Categorical columns to encode: {categorical_cols_to_encode}")
    for col in categorical_cols_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    print("DataFrame after encoding categorical features:")
    print(df.head())
    return df

def scale_numerical_features(df, target_column='is_fraud'):
    """Applies StandardScaler to numerical features, excluding the target column."""
    numerical_cols_to_scale = df.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    if target_column in numerical_cols_to_scale:
        numerical_cols_to_scale.remove(target_column)

    # If 'amt_winsorized' is present and preferred over original 'amt', remove 'amt'
    if 'amt_winsorized' in numerical_cols_to_scale and 'amt' in numerical_cols_to_scale:
        numerical_cols_to_scale.remove('amt')

    print(f"Numerical columns to scale: {numerical_cols_to_scale}")
    scaler = StandardScaler()
    df[numerical_cols_to_scale] = scaler.fit_transform(df[numerical_cols_to_scale])
    print("DataFrame after scaling numerical features:")
    print(df.head())
    return df, scaler

def bin_features(df, original_df):
    """Bins 'amt_winsorized' and 'age' into categorical features."""
    # Binning 'amt_winsorized' into categories
    df['amt_bin'] = pd.cut(df['amt_winsorized'], bins=5, labels=False, include_lowest=True)
    print("\nDistribution of 'amt_bin':")
    print(df['amt_bin'].value_counts().sort_index())

    # Binning 'age' into categories - Using the *original* age values for binning for correct interpretation
    # Recalculate original age for binning purposes using the original 'df'
    original_dob = pd.to_datetime(original_df['dob'])
    original_trans_date_trans_time = pd.to_datetime(original_df['trans_date_trans_time'])
    original_age_for_binning = original_trans_date_trans_time.dt.year - original_dob.dt.year

    # Define age bins and labels
    age_bins = [0, 25, 45, 65, np.inf] # Define age ranges
    age_labels = ['<25', '25-45', '45-65', '65+']
    df['age_bin'] = pd.cut(original_age_for_binning, bins=age_bins, labels=age_labels, right=False)
    print("\nDistribution of 'age_bin':")
    print(df['age_bin'].value_counts().sort_index())

    print("\nDataFrame head with new binned features:")
    print(df[['amt_winsorized', 'amt_bin', 'age', 'age_bin']].head())
    return df
