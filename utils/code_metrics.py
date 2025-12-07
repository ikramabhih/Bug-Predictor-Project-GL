import ast
import re
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze
import numpy as np

def extract_metrics_from_code(code, filename="code.py"):
    """
    Extrait les métriques de code pour la prédiction avec validation stricte
    """
    try:
        # Vérifier que le code n'est pas vide
        if not code or len(code.strip()) == 0:
            print(f"Empty code provided for {filename}")
            return create_minimal_metrics()
        
        # Tenter de parser le code pour vérifier la syntaxe
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            print(f"Syntax error in {filename}: {e}")
            # Retourner des métriques minimales au lieu de None
            return create_minimal_metrics()
        
        # Analyse de base avec radon
        raw_metrics = analyze(code)
        
        # CHANGEMENT: Accepter même le code simple
        # On ne retourne plus None pour les petits fichiers
        if raw_metrics.loc < 3:
            print(f"Simple code in {filename} (LOC={raw_metrics.loc}) - using minimal metrics")
            return create_minimal_metrics(raw_metrics.loc)
        
        # Complexité cyclomatique
        complexity_blocks = cc_visit(code)
        if complexity_blocks:
            total_complexity = sum(block.complexity for block in complexity_blocks)
            avg_complexity = total_complexity / len(complexity_blocks)
            max_complexity = max(block.complexity for block in complexity_blocks)
            branch_count = len(complexity_blocks)
        else:
            total_complexity = 1
            avg_complexity = 1
            max_complexity = 1
            branch_count = 0
        
        # Métriques Halstead
        try:
            halstead = h_visit(code)
        except:
            halstead = None
        
        # Compter les opérateurs et opérandes via AST
        operators = count_operators(tree)
        operands = count_operands(tree)
        
        # Calcul de la complexité essentielle
        ev_g = calculate_essential_complexity(tree)
        
        # Calcul de la complexité de design
        iv_g = calculate_design_complexity(tree)
        
        # Validation des valeurs Halstead
        halstead_volume = halstead.total.volume if halstead else max(raw_metrics.loc * 2, 1)
        halstead_difficulty = halstead.total.difficulty if halstead else max(1, operators['unique'] / max(operands['unique'], 1))
        halstead_effort = halstead.total.effort if halstead else halstead_volume * halstead_difficulty
        halstead_time = halstead.total.time if halstead else halstead_effort / 18
        halstead_bugs = halstead.total.bugs if halstead else halstead_volume / 3000
        halstead_length = halstead.total.length if halstead else operators['total'] + operands['total']
        
        # S'assurer que toutes les valeurs sont finies
        def safe_value(value, default=1):
            if value is None or np.isnan(value) or np.isinf(value):
                return default
            return max(value, 0)
        
        # Métriques finales (21 features attendues)
        metrics = {
            'loc': safe_value(raw_metrics.loc, 1),
            'v(g)': safe_value(total_complexity, 1),
            'ev(g)': safe_value(ev_g, 1),
            'iv(g)': safe_value(iv_g, 1),
            'n': safe_value(halstead_length, 1),
            'v': safe_value(halstead_volume, 1),
            'l': safe_value(1.0 / max(halstead_difficulty, 0.001), 0.1),
            'd': safe_value(halstead_difficulty, 1),
            'i': safe_value(halstead_volume / max(halstead_difficulty, 0.001), 1),
            'e': safe_value(halstead_effort, 1),
            'b': safe_value(halstead_bugs, 0.001),
            't': safe_value(halstead_time, 0.1),
            'lOCode': safe_value(raw_metrics.loc, 1),
            'lOComment': safe_value(raw_metrics.comments, 0),
            'lOBlank': safe_value(raw_metrics.blank, 0),
            'locCodeAndComment': safe_value(raw_metrics.multi, 0),
            'uniq_Op': safe_value(operators['unique'], 1),
            'uniq_Opnd': safe_value(operands['unique'], 1),
            'total_Op': safe_value(operators['total'], 1),
            'total_Opnd': safe_value(operands['total'], 1),
            'branchCount': safe_value(branch_count, 0)
        }
        
        # Validation finale
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                print(f"Invalid metric {key}={value} in {filename}, using default")
                metrics[key] = 1 if key != 'b' else 0.001
        
        return metrics
        
    except Exception as e:
        print(f"Error extracting metrics from {filename}: {e}")
        return create_minimal_metrics()


def create_minimal_metrics(loc=5):
    """
    Crée un ensemble minimal de métriques valides pour les cas d'erreur
    Cela évite de retourner None et de faire échouer l'analyse
    """
    return {
        'loc': loc,
        'v(g)': 1,
        'ev(g)': 1,
        'iv(g)': 1,
        'n': 10,
        'v': 20,
        'l': 0.5,
        'd': 2,
        'i': 10,
        'e': 40,
        'b': 0.01,
        't': 2.2,
        'lOCode': loc,
        'lOComment': 0,
        'lOBlank': 1,
        'locCodeAndComment': 0,
        'uniq_Op': 3,
        'uniq_Opnd': 5,
        'total_Op': 5,
        'total_Opnd': 8,
        'branchCount': 1
    }


def calculate_essential_complexity(tree):
    """Calcule la complexité essentielle"""
    complexity = 1
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            complexity += 1
        elif isinstance(node, ast.If):
            complexity += 1
        elif isinstance(node, ast.Try):
            complexity += len(node.handlers)
        elif isinstance(node, ast.With):
            complexity += 1
    
    return max(complexity, 1)


def calculate_design_complexity(tree):
    """Calcule la complexité de design"""
    complexity = 1
    
    function_defs = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    function_calls = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call))
    imports = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
    
    complexity = max(function_defs + function_calls * 0.5 + imports * 0.3, 1)
    
    return complexity


def count_operators(tree):
    """Compte les opérateurs dans l'AST"""
    operators = set()
    total = 0
    
    operator_types = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
        ast.And, ast.Or, ast.Not,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.BitAnd, ast.BitOr, ast.BitXor, ast.Invert, ast.LShift, ast.RShift,
        ast.UAdd, ast.USub,
        ast.In, ast.NotIn, ast.Is, ast.IsNot
    )
    
    for node in ast.walk(tree):
        if isinstance(node, operator_types):
            operators.add(type(node).__name__)
            total += 1
        elif isinstance(node, ast.Assign):
            operators.add('Assign')
            total += 1
    
    return {
        'unique': max(len(operators), 1),
        'total': max(total, 1)
    }


def count_operands(tree):
    """Compte les opérandes dans l'AST"""
    operands = set()
    total = 0
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            operands.add(f"var_{node.id}")
            total += 1
        elif isinstance(node, ast.Constant):
            operands.add(f"const_{type(node.value).__name__}_{str(node.value)[:20]}")
            total += 1
        elif isinstance(node, ast.Attribute):
            operands.add(f"attr_{node.attr}")
            total += 1
    
    return {
        'unique': max(len(operands), 1),
        'total': max(total, 1)
    }


def extract_metrics_from_file(filepath):
    """Extrait les métriques depuis un fichier avec gestion des encodages"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                code = f.read()
            return extract_metrics_from_code(code, filepath)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file {filepath} with {encoding}: {e}")
            continue
    
    print(f"Failed to read {filepath} with any encoding")
    return create_minimal_metrics()


def compare_with_thresholds(metrics):
    """Compare les métriques avec des seuils standards"""
    warnings = []
    
    if metrics['loc'] > 300:
        warnings.append('Code très long (>300 LOC)')
    
    if metrics['v(g)'] > 15:
        warnings.append('Complexité cyclomatique élevée (>15)')
    
    if metrics['branchCount'] > 20:
        warnings.append('Trop de branches (>20)')
    
    if metrics['lOComment'] / max(metrics['loc'], 1) < 0.1:
        warnings.append('Manque de commentaires (<10%)')
    
    return warnings