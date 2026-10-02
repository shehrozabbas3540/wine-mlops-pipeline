import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def load_and_split_data():
    wine = load_wine(as_frame=True)
    df = wine.frame
    
    assert df.isnull().sum().sum() == 0, "Error: Dataset contains null values"
    assert len(wine.feature_names) == 13, "Error: Feature count is not exactly 13"
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"Success! Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")