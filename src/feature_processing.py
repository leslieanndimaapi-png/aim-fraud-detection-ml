
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """Splits the data into training and testing sets, optionally stratified."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")
    print("
Distribution of 'is_fraud' in y_train:")
    print(y_train.value_counts(normalize=True))
    print("
Distribution of 'is_fraud' in y_test:")
    print(y_test.value_counts(normalize=True))
    return X_train, X_test, y_train, y_test

def select_features(X_train, y_train, X_test, k_features=15):
    """Applies SelectKBest to select top features and transforms the datasets."""
    selector = SelectKBest(score_func=f_classif, k=k_features)
    selector.fit(X_train, y_train)

    selected_features_mask = selector.get_support()
    selected_feature_names = X_train.columns[selected_features_mask]

    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Selected number of features: {X_train_selected.shape[1]}")
    print(f"
Top {k_features} selected features based on f_classif score:")
    print(selected_feature_names)

    X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_feature_names, index=X_train.index)
    X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_feature_names, index=X_test.index)

    print("
Head of X_train_selected_df:")
    print(X_train_selected_df.head())

    return X_train_selected_df, X_test_selected_df, selector, selected_feature_names
