import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_and_preprocess_data(filepath):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Strip whitespace from column names to prevent matching errors
    df.columns = df.columns.str.strip()
    
    target_column = 'Life expectancy'
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}")
        
    print(f"[INFO] Target column '{target_column}' found successfully.")
    
    # Handle missing values by dropping rows where target is NaN
    df = df.dropna(subset=[target_column])
    
    # Encode categorical 'Status' column if present ('Developed' / 'Developing')
    if 'Status' in df.columns:
        df['Status'] = df['Status'].map({'Developed': 1, 'Developing': 0})
        
    # Select numeric features for prediction
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fill missing values in features with median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Define features (X) and target (y)
    X = df.drop(columns=['Country', target_column], errors='ignore')
    y = df[target_column]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data('life_expectancy_data.csv')
        print(f"[INFO] Training set shape: {X_train.shape}")
        print(f"[INFO] Testing set shape: {X_test.shape}")
        
        print("[INFO] Training Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        print("[INFO] Evaluating model...")
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("--- Model Performance ---")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R-squared (R2 Score): {r2:.4f}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
