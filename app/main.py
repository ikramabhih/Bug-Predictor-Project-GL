import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.train import train_model
from pipeline.predict import predict
from core.random_forest import RandomForestModel
from core.xgboost_model import XGBoostModel
from core.logistic_regression import LogisticRegressionModel

def main():
    parser = argparse.ArgumentParser(description="Bug Predictor App")
    parser.add_argument("--mode", choices=["train", "predict"], required=True,
                        help="Mode: train or predict")
    parser.add_argument("--model", choices=["rf", "xgb", "lr"], default="rf",
                        help="Model: rf (Random Forest), xgb (XGBoost), lr (Logistic Regression)")
    
    args = parser.parse_args()
    
    # Sélectionner le modèle
    models = {
        "rf": RandomForestModel(),
        "xgb": XGBoostModel(),
        "lr": LogisticRegressionModel()
    }
    
    model = models[args.model]
    model_name = args.model
    
    if args.mode == "train":
        print(f"\n Training {model_name.upper()} model...")
        train_model(model, 
                   model_path=f"data/model_{model_name}.joblib",
                   scaler_path=f"data/scaler_{model_name}.joblib")
    elif args.mode == "predict":
        print(f"\n Predicting with {model_name.upper()} model...")
        predict(model_path=f"data/model_{model_name}.joblib",
               scaler_path=f"data/scaler_{model_name}.joblib")

if __name__ == "__main__":
    main()