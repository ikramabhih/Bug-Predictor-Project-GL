import sys
import os
import joblib
import json

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import load_data, prepare_data, scale_data
from utils.metrics import compute_metrics

def train_model(model, model_path="data/model.joblib", scaler_path="data/scaler.joblib", data_path="data/cleaned_data.csv"):
    """
    Entraîne un modèle et sauvegarde les artifacts
    
    Args:
        model: Instance du modèle à entraîner
        model_path: Chemin pour sauvegarder le modèle (ex: data/model_rf.joblib)
        scaler_path: Chemin pour sauvegarder le scaler (ex: data/scaler_rf.joblib)
        data_path: Chemin vers les données d'entraînement
    """
    # Extraire le type de modèle depuis le nom du fichier
    model_type = "unknown"
    if "rf" in model_path:
        model_type = "rf"
    elif "xgb" in model_path:
        model_type = "xgb"
    elif "lr" in model_path:
        model_type = "lr"
    
    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*50}\n")
    
    print(" Loading data...")
    try:
        df = load_data(data_path)
        print(f"Loaded {len(df)} samples")
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        print(f"   Please run preprocessing first:")
        print(f"   python -m utils.preprocessing")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("\nPreparing data...")
    try:
        X_train, X_test, y_train, y_test = prepare_data(df)
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"   Class distribution (train): {sum(y_train)} bugs / {len(y_train)} total")
    except Exception as e:
        print(f"Error preparing data: {e}")
        return
    
    print("\n⚖️  Scaling data...")
    try:
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        print("Data scaled successfully")
    except Exception as e:
        print(f"Error scaling data: {e}")
        return
    
    print(f"\n🔧 Training {type(model).__name__}...")
    try:
        model.fit(X_train_scaled, y_train)
        print("Model trained successfully")
    except Exception as e:
        print(f"Error training model: {e}")
        return
    
    print("\n Evaluating model...")
    try:
        y_pred = model.predict(X_test_scaled)
        
        # Vérifier si le modèle a predict_proba
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
            y_proba = model.model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = y_pred  # Fallback
        
        metrics = compute_metrics(y_test, y_pred, y_proba)
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE")
        print("="*50)
        for metric_name, value in metrics.items():
            print(f"  {metric_name:20s}: {value:.4f}")
        print("="*50 + "\n")
    except Exception as e:
        print(f" Error evaluating model: {e}")
        metrics = {}
    
    print("Saving artifacts...")
    try:
        # Créer le dossier data s'il n'existe pas
        os.makedirs("data", exist_ok=True)
        
        # Sauvegarder le modèle et le scaler
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        
        # Sauvegarder les métriques dans un fichier JSON
        metrics_file = "data/model_metrics.json"
        all_metrics = {}
        
        # Charger les métriques existantes si le fichier existe
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    all_metrics = json.load(f)
            except:
                all_metrics = {}
        
        # Ajouter les nouvelles métriques
        all_metrics[model_type] = metrics
        
        # Sauvegarder
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        print(f"Metrics saved to {metrics_file}")
        
    except Exception as e:
        print(f"Error saving artifacts: {e}")
        return
    
    print("\nTraining completed successfully!\n")
    return metrics


if __name__ == "__main__":
    """
    Exemple d'utilisation directe
    """
    from core.random_forest import RandomForestModel
    
    print("Starting model training...")
    model = RandomForestModel()
    train_model(
        model, 
        model_path="data/model_rf.joblib",
        scaler_path="data/scaler_rf.joblib"
    )