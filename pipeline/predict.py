import sys
import os
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def predict(data_path="data/cleaned_data.csv", model_path="data/model.joblib", 
            scaler_path="data/scaler.joblib", threshold=0.5):
    print("Loading model and scaler...")
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print("Model or scaler not found. Please train the model first.")
        return

    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return
    
    if "defects" in df.columns:
        X = df.drop("defects", axis=1)
        y_true = df["defects"]
    else:
        X = df
        y_true = None
    
    print("Scaling input...")
    X_scaled = scaler.transform(X)
    
    print(f"Predicting with threshold={threshold}...")
    
    # Utiliser predict_proba si disponible
    if hasattr(model, 'model'):
        probas = model.model.predict_proba(X_scaled)[:, 1]
    else:
        probas = model.predict_proba(X_scaled)[:, 1]
    
    predictions = (probas >= threshold).astype(int)
    
    results = pd.DataFrame({
        "Predicted": predictions,
        "Probability": probas
    })
    
    if y_true is not None:
        results["Actual"] = y_true
        
        from utils.metrics import compute_metrics
        metrics = compute_metrics(y_true, predictions)
        
        print(f"\nMetrics (threshold={threshold}):")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1 Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    
    print("\nPredictions (first 10 rows):")
    print(results.head(10))

if __name__ == "__main__":
    predict()