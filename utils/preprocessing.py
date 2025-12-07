import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path="data/cleaned_data.csv"):
    df = pd.read_csv(path)
    return df

def prepare_data(df):
    X = df.drop("defects", axis=1)
    y = df["defects"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler