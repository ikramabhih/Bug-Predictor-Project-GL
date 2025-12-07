from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from git import Repo

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.random_forest import RandomForestModel
from core.xgboost_model import XGBoostModel
from core.logistic_regression import LogisticRegressionModel
from utils.code_metrics import extract_metrics_from_code

app = Flask(__name__)
CORS(app)

MODELS = {}
SCALERS = {}

# Seuils de détection
THRESHOLDS = {
    'rf': 0.5,   
    'xgb': 0.5,  
    'lr': 0.5    
}

def load_models():
    """Charge tous les modèles disponibles"""
    model_types = ['rf', 'xgb', 'lr']
    for model_type in model_types:
        model_path = f'data/model_{model_type}.joblib'
        scaler_path = f'data/scaler_{model_type}.joblib'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                MODELS[model_type] = joblib.load(model_path)
                SCALERS[model_type] = joblib.load(scaler_path)
                print(f"✓ Loaded {model_type.upper()} model")
            except Exception as e:
                print(f"✗ Error loading {model_type}: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'models': list(MODELS.keys()),
        'thresholds': THRESHOLDS,
        'status': 'success'
    })

def validate_metrics(metrics):
    """Valide que les métriques extraites sont réalistes"""
    required_features = [
        'loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e',
        'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment',
        'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount'
    ]
    
    for feature in required_features:
        if feature not in metrics:
            raise ValueError(f"Missing feature: {feature}")
        if not isinstance(metrics[feature], (int, float)):
            raise ValueError(f"Invalid type for {feature}")
        if metrics[feature] < 0:
            raise ValueError(f"Negative value for {feature}")
    
    return True

def prepare_features(metrics):
    """Prépare les features pour la prédiction avec validation"""
    feature_names = [
        'loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e',
        'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment',
        'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount'
    ]
    
    feature_values = []
    for name in feature_names:
        value = metrics.get(name, 0)
        # Éviter les valeurs infinies ou NaN
        if np.isnan(value) or np.isinf(value):
            value = 0
        feature_values.append(float(value))
    
    return pd.DataFrame([feature_values], columns=feature_names)

def get_prediction(model, scaler, X, model_type):
    """Effectue la prédiction avec gestion des cas limites"""
    X_scaled = scaler.transform(X)
    
    if hasattr(model, 'model'):
        proba = model.model.predict_proba(X_scaled)[0]
    else:
        proba = model.predict_proba(X_scaled)[0]
    
    # Utiliser le seuil spécifique au modèle
    threshold = THRESHOLDS.get(model_type, 0.5)
    prediction = 1 if proba[1] >= threshold else 0
    
    return prediction, proba

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_type = data.get('model', 'rf')
        features = data.get('features', {})
        
        if model_type not in MODELS:
            return jsonify({
                'error': f'Model {model_type} not found',
                'available_models': list(MODELS.keys()),
                'status': 'error'
            }), 404
        
        # Valider les métriques
        validate_metrics(features)
        
        # Préparer les features
        X = prepare_features(features)
        
        # Prédire
        prediction, probability = get_prediction(
            MODELS[model_type], 
            SCALERS[model_type], 
            X, 
            model_type
        )
        
        # Déterminer le niveau de risque
        bug_proba = float(probability[1])
        if bug_proba > 0.7:
            risk_level = 'high'
        elif bug_proba > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return jsonify({
            'prediction': int(prediction),
            'probability': {
                'no_bug': float(probability[0]),
                'bug': bug_proba
            },
            'risk_level': risk_level,
            'threshold_used': THRESHOLDS[model_type],
            'model_used': model_type,
            'status': 'success'
        })
        
    except ValueError as e:
        return jsonify({
            'error': f'Invalid metrics: {str(e)}',
            'status': 'error'
        }), 400
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/analyze/code', methods=['POST'])
def analyze_code():
    try:
        data = request.get_json()
        model_type = data.get('model', 'rf')
        code = data.get('code', '')
        
        if not code or len(code.strip()) == 0:
            return jsonify({
                'error': 'No code provided or code is empty',
                'status': 'error'
            }), 400
        
        if model_type not in MODELS:
            return jsonify({
                'error': f'Model {model_type} not found',
                'status': 'error'
            }), 404
        
        # Extraire les métriques
        metrics = extract_metrics_from_code(code)
        
        # Vérifier si l'extraction a échoué
        if metrics is None:
            return jsonify({
                'error': 'Failed to extract metrics from code',
                'status': 'error'
            }), 400
        
        # Valider les métriques
        try:
            validate_metrics(metrics)
        except ValueError as e:
            return jsonify({
                'error': f'Invalid metrics extracted: {str(e)}',
                'metrics': metrics,
                'status': 'error'
            }), 400
        
        # Préparer et prédire
        X = prepare_features(metrics)
        prediction, probability = get_prediction(
            MODELS[model_type], 
            SCALERS[model_type], 
            X, 
            model_type
        )
        
        bug_proba = float(probability[1])
        if bug_proba > 0.7:
            risk_level = 'high'
        elif bug_proba > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return jsonify({
            'prediction': int(prediction),
            'probability': {
                'no_bug': float(probability[0]),
                'bug': bug_proba
            },
            'risk_level': risk_level,
            'threshold_used': THRESHOLDS[model_type],
            'model_used': model_type,
            'metrics': metrics,
            'code_length': len(code),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': str(e.__traceback__),
            'status': 'error'
        }), 500

@app.route('/api/analyze/files', methods=['POST'])
def analyze_files():
    try:
        model_type = request.form.get('model', 'rf')
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({'error': 'No files provided', 'status': 'error'}), 400
        
        if model_type not in MODELS:
            return jsonify({'error': f'Model {model_type} not found', 'status': 'error'}), 404
        
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            temp_path = os.path.join(tempfile.gettempdir(), file.filename)
            file.save(temp_path)
            
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                metrics = extract_metrics_from_code(code, file.filename)
                
                if metrics is None:
                    results.append({
                        'file': file.filename,
                        'prediction': -1,
                        'bug_probability': 0.0,
                        'risk_level': 'error',
                        'error': 'Failed to extract metrics'
                    })
                    continue
                
                validate_metrics(metrics)
                X = prepare_features(metrics)
                prediction, probability = get_prediction(
                    MODELS[model_type], 
                    SCALERS[model_type], 
                    X, 
                    model_type
                )
                
                bug_proba = float(probability[1])
                results.append({
                    'file': file.filename,
                    'prediction': int(prediction),
                    'bug_probability': bug_proba,
                    'risk_level': 'high' if bug_proba > 0.7 else 'medium' if bug_proba > 0.4 else 'low',
                    'metrics': {
                        'loc': metrics.get('loc', 0),
                        'complexity': metrics.get('v(g)', 0),
                        'branches': metrics.get('branchCount', 0)
                    }
                })
                
            except Exception as e:
                results.append({
                    'file': file.filename,
                    'prediction': -1,
                    'bug_probability': 0.0,
                    'risk_level': 'error',
                    'error': str(e)
                })
            
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        results.sort(key=lambda x: x.get('bug_probability', 0), reverse=True)
        
        return jsonify({
            'results': results,
            'total_files': len(results),
            'high_risk_count': sum(1 for r in results if r.get('risk_level') == 'high'),
            'model_used': model_type,
            'threshold_used': THRESHOLDS[model_type],
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/analyze/git', methods=['POST'])
def analyze_git():
    try:
        data = request.get_json()
        model_type = data.get('model', 'rf')
        git_url = data.get('git_url', '').strip()
        
        if not git_url:
            return jsonify({'error': 'No Git URL provided', 'status': 'error'}), 400
        
        # Valider l'URL
        if not (git_url.startswith('https://') or git_url.startswith('http://')):
            return jsonify({'error': 'Invalid Git URL. Must start with https:// or http://', 'status': 'error'}), 400
        
        if model_type not in MODELS:
            return jsonify({'error': f'Model {model_type} not found', 'status': 'error'}), 404
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            print(f"Cloning repository: {git_url}")
            
            # Configuration pour éviter les problèmes d'authentification
            import os
            env = os.environ.copy()
            env['GIT_TERMINAL_PROMPT'] = '0'  # Désactiver les prompts d'authentification
            
            # Cloner sans authentification (pour les repos publics)
            try:
                # Méthode 1: Clone simple avec GitPython
                Repo.clone_from(
                    git_url, 
                    temp_dir,
                    env=env,
                    no_checkout=False,
                    depth=1  # Clone shallow pour être plus rapide
                )
            except Exception as clone_error:
                # Méthode 2: Utiliser subprocess comme fallback
                import subprocess
                print(f"GitPython failed, trying subprocess: {clone_error}")
                
                result = subprocess.run(
                    ['git', 'clone', '--depth', '1', git_url, temp_dir],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=300  # 5 minutes max
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    
                    # Messages d'erreur plus clairs
                    if 'not found' in error_msg.lower():
                        return jsonify({
                            'error': 'Repository not found. Please check the URL.',
                            'details': 'Make sure the repository exists and is public.',
                            'url': git_url,
                            'status': 'error'
                        }), 404
                    elif 'authentication' in error_msg.lower() or 'credential' in error_msg.lower():
                        return jsonify({
                            'error': 'This repository requires authentication.',
                            'details': 'Please use a public repository URL.',
                            'status': 'error'
                        }), 403
                    else:
                        return jsonify({
                            'error': 'Failed to clone repository',
                            'details': error_msg[:500],  # Limiter la longueur
                            'status': 'error'
                        }), 500
            
            print(f"Repository cloned successfully to {temp_dir}")
            
            results = []
            total_files = 0
            skipped_files = 0
            
            # Extensions de fichiers supportés
            supported_extensions = ('.py', '.java', '.js', '.cpp', '.c', '.cs', '.php', '.rb', '.go')
            
            for root, dirs, files in os.walk(temp_dir):
                # Ignorer les dossiers cachés et les dossiers de build
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env', 'build', 'dist']]
                
                for file in files:
                    if file.endswith(supported_extensions):
                        total_files += 1
                        filepath = os.path.join(root, file)
                        relative_path = os.path.relpath(filepath, temp_dir)
                        
                        # Limiter à 100 fichiers pour éviter les timeouts
                        if len(results) >= 100:
                            skipped_files = total_files - len(results)
                            break
                        
                        try:
                            # Lire le fichier avec gestion des encodages
                            code = None
                            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                                try:
                                    with open(filepath, 'r', encoding=encoding) as f:
                                        code = f.read()
                                    break
                                except UnicodeDecodeError:
                                    continue
                            
                            if code is None:
                                print(f"Could not read {relative_path} with any encoding")
                                continue
                            
                            # Extraire les métriques
                            metrics = extract_metrics_from_code(code, relative_path)
                            
                            if metrics is None:
                                continue
                            
                            # Valider et prédire
                            validate_metrics(metrics)
                            X = prepare_features(metrics)
                            prediction, probability = get_prediction(
                                MODELS[model_type], 
                                SCALERS[model_type], 
                                X, 
                                model_type
                            )
                            
                            bug_proba = float(probability[1])
                            results.append({
                                'file': relative_path,
                                'prediction': int(prediction),
                                'bug_probability': bug_proba,
                                'risk_level': 'high' if bug_proba > 0.7 else 'medium' if bug_proba > 0.4 else 'low',
                                'metrics': {
                                    'loc': metrics.get('loc', 0),
                                    'complexity': metrics.get('v(g)', 0),
                                    'branches': metrics.get('branchCount', 0)
                                }
                            })
                            
                        except Exception as e:
                            print(f"Error analyzing {relative_path}: {e}")
                            continue
                
                if len(results) >= 100:
                    break
            
            # Trier par probabilité de bug (plus élevé en premier)
            results.sort(key=lambda x: x['bug_probability'], reverse=True)
            
            response_data = {
                'results': results,
                'total_files_analyzed': len(results),
                'total_files_found': total_files,
                'skipped_files': skipped_files,
                'high_risk_count': sum(1 for r in results if r['risk_level'] == 'high'),
                'medium_risk_count': sum(1 for r in results if r['risk_level'] == 'medium'),
                'low_risk_count': sum(1 for r in results if r['risk_level'] == 'low'),
                'model_used': model_type,
                'threshold_used': THRESHOLDS[model_type],
                'repository_url': git_url,
                'status': 'success'
            }
            
            if skipped_files > 0:
                response_data['warning'] = f'Analysis limited to first 100 files. {skipped_files} files were skipped.'
            
            return jsonify(response_data)
            
        except subprocess.TimeoutExpired:
            return jsonify({
                'error': 'Repository clone timeout',
                'details': 'The repository took too long to clone (>5 minutes). Try a smaller repository.',
                'status': 'error'
            }), 408
            
        finally:
            # Nettoyer le dossier temporaire
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
        
    except Exception as e:
        import traceback
        print(f"Error in analyze_git: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'status': 'error'
        }), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics_file = 'data/model_metrics.json'
    if os.path.exists(metrics_file):
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return jsonify({
            'metrics': metrics,
            'thresholds': THRESHOLDS,
            'status': 'success'
        })
    else:
        return jsonify({
            'error': 'Metrics not found. Train models first.',
            'status': 'error'
        }), 404

@app.route('/api/thresholds', methods=['POST'])
def update_thresholds():
    """Met à jour les seuils de détection"""
    try:
        data = request.get_json()
        for model_type, threshold in data.items():
            if model_type in MODELS and 0 <= threshold <= 1:
                THRESHOLDS[model_type] = threshold
        
        return jsonify({
            'thresholds': THRESHOLDS,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print("Starting Bug Predictor API...")
    load_models()
    
    if not MODELS:
        print("\nWARNING: No models loaded!")
        print("   Please train models first:")
        print("   python app/main.py --mode train --model rf")
        print("   python app/main.py --mode train --model xgb")
        print("   python app/main.py --mode train --model lr\n")
    else:
        print(f"\n Loaded {len(MODELS)} model(s)")
        print(f" Current thresholds: {THRESHOLDS}\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)