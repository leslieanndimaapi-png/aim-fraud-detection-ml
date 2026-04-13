
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.utils import class_weight

def get_class_weights(y_train):
    """Calculates class weights for imbalanced datasets."""
    class_weights_array = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights_array))
    print(f"Calculated class weights: {class_weight_dict}")
    return class_weight_dict

def initialize_logistic_regression(random_state=42, class_weight_dict=None):
    """Initializes a Logistic Regression model."""
    cw = 'balanced' if class_weight_dict else None
    model = LogisticRegression(solver='liblinear', class_weight=cw, random_state=random_state, n_jobs=-1)
    print("Initialized Logistic Regression model.")
    return model

def initialize_decision_tree(random_state=42, class_weight_dict=None):
    """Initializes a Decision Tree Classifier model."""
    cw = 'balanced' if class_weight_dict else None
    model = DecisionTreeClassifier(class_weight=cw, random_state=random_state)
    print("Initialized Decision Tree Classifier model.")
    return model

def initialize_random_forest(random_state=42, class_weight_dict=None):
    """Initializes a Random Forest Classifier model."""
    cw = 'balanced' if class_weight_dict else None
    model = RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=random_state, n_jobs=-1)
    print("Initialized Random Forest Classifier model.")
    return model

def initialize_xgboost(random_state=42, scale_pos_weight_value=None):
    """Initializes an XGBoost Classifier model."""
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False, # Suppress warning
        scale_pos_weight=scale_pos_weight_value, # Only if not SMOTE-balanced
        random_state=random_state,
        n_jobs=-1,
        enable_categorical=True # Enable categorical feature support
    )
    print("Initialized XGBoost Classifier model.")
    return model

def initialize_linear_svc(random_state=42, class_weight_dict=None):
    """Initializes a Linear SVM Classifier model."""
    cw = 'balanced' if class_weight_dict else None
    model = LinearSVC(
        class_weight=cw,
        random_state=random_state,
        dual=False, # Use the primal formulation for large number of samples
        max_iter=2000 # Increase max iterations for convergence
    )
    print("Initialized LinearSVC model.")
    return model

def train_model(model, X_train, y_train, model_name="Model"): # Removed early_stopping_rounds and callbacks to simplify for initial refactor
    """Trains a given machine learning model."""
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    print(f"{model_name} training complete.")
    return model

def train_xgboost_with_validation(model, X_train, y_train, X_val, y_val, model_name="XGBoost Model"):
    """Trains an XGBoost model with a validation set for early stopping."""
    print(f"Training {model_name} with validation set...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False # Set to True for verbose output during training
    )
    print(f"{model_name} training complete.")
    return model
