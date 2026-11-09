import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def load_and_preprocess_data(file_path):
    """Loads the dataset, handles missing values, and splits features/target."""
    print("[INFO] Loading dataset...")
    # Replace 'life_expectancy_data.csv' with your actual WHO dataset file path
    df = pd.read_csv(file_path)
    
    # Assuming 'Life expectancy' is the target column and others are features
    # Adjust column names based on your specific CSV file from Kaggle/WHO
    target_column = 'Life expectancy'
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical variables if any exist
    X = pd.get_dummies(X, drop_first=True)

    # Train-Test Split (80% training, 20% testing) as per Chapter 3.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handling missing values using median imputation
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Feature scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Trains multiple models and evaluates them using MAE, MSE, and R2 Score."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"[INFO] Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation metrics specified in synopsis
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "R2 Score": r2
        })

    results_df = pd.DataFrame(results)
    print("\n--- Model Evaluation Results ---")
    print(results_df.to_string(index=False))
    return results_df

if __name__ == "__main__":
    try:
        # Update path to your dataset CSV file
        DATASET_PATH = "life_expectancy_data.csv"
        
        X_train, X_test, y_train, y_test = load_and_preprocess_data(DATASET_PATH)
        train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
    except FileNotFoundError:
        print(f"[ERROR] Could not find '{DATASET_PATH}'. Please place your CSV file in the project directory.")
