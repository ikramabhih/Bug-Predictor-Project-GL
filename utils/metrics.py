"""
Fonctions pour calculer les métriques de performance des modèles
"""
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import numpy as np

def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Calcule les métriques de performance du modèle
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions du modèle
        y_proba: Probabilités prédites (optionnel, pour AUC-ROC)
    
    Returns:
        dict: Dictionnaire contenant toutes les métriques
    """
    # Convertir en numpy arrays si ce sont des Series pandas
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_proba is not None:
        y_proba = np.array(y_proba)
    metrics = {}
    
    # Métriques de base
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC-ROC si les probabilités sont disponibles
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = 0.0
    
    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    # Spécificité
    if (tn + fp) > 0:
        metrics['specificity'] = tn / (tn + fp)
    else:
        metrics['specificity'] = 0.0
    
    return metrics


def print_metrics(metrics):
    """
    Affiche les métriques de manière formatée
    
    Args:
        metrics: Dictionnaire de métriques
    """
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    
    # Métriques principales
    print("\nMain Metrics:")
    print(f"  Accuracy:    {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision:   {metrics.get('precision', 0):.4f}")
    print(f"  Recall:      {metrics.get('recall', 0):.4f}")
    print(f"  F1-Score:    {metrics.get('f1_score', 0):.4f}")
    
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:     {metrics.get('roc_auc', 0):.4f}")
    
    # Matrice de confusion
    print("\n Confusion Matrix:")
    print(f"  True Positives:  {metrics.get('true_positives', 0)}")
    print(f"  True Negatives:  {metrics.get('true_negatives', 0)}")
    print(f"  False Positives: {metrics.get('false_positives', 0)}")
    print(f"  False Negatives: {metrics.get('false_negatives', 0)}")
    print(f"  Specificity:     {metrics.get('specificity', 0):.4f}")
    
    print("="*50 + "\n")


def compare_models(metrics_dict):
    """
    Compare plusieurs modèles
    
    Args:
        metrics_dict: Dictionnaire {nom_modele: metrics}
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    
    for model_name, metrics in metrics_dict.items():
        print(f"{model_name:<15} "
              f"{metrics.get('accuracy', 0):<12.4f} "
              f"{metrics.get('precision', 0):<12.4f} "
              f"{metrics.get('recall', 0):<12.4f} "
              f"{metrics.get('f1_score', 0):<12.4f}")
    
    print("="*70 + "\n")