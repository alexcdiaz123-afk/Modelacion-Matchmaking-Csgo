"""
Módulo de funciones utilitarias para estadística, serialización JSON y formateo.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats


class NumpyEncoder(json.JSONEncoder):
    """Codificador JSON personalizado para manejar tipos de NumPy, pandas y valores especiales."""
    
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)


def safe_float(value, decimals=4):
    """Convierte un valor a float redondeado de forma segura."""
    try:
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return None
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    """Convierte un valor a int de forma segura."""
    try:
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def format_params(params):
    """Formatea los parámetros de una distribución para visualización académica."""
    if not params:
        return "N/A"
    
    param_names = {
        'lambda': 'λ',
        'mu': 'μ',
        'sigma': 'σ',
        'shape': 'α (forma)',
        'scale': 'β (escala)',
        'n': 'n',
        'p': 'p',
        'r': 'r',
        'a': 'a',
        'b': 'b',
        'dispersion_ratio': 'Var/Media'
    }
    
    parts = []
    for key, value in params.items():
        name = param_names.get(key, key)
        val_str = f"{safe_float(value, 4)}" if isinstance(value, (int, float, np.number)) else str(value)
        parts.append(f"{name} = {val_str}")
        
    return ", ".join(parts)


def calculate_descriptive_stats(data):
    """
    Calcula un conjunto exhaustivo de estadísticas descriptivas para una muestra.
    
    Args:
        data: Array o serie numérica.
        
    Returns:
        Diccionario con estadísticos de tendencia central, dispersión y forma.
    """
    arr = np.array(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    
    if len(arr) == 0:
        return {}
        
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    n = len(arr)
    mean_val = float(np.mean(arr))
    var_val = float(np.var(arr, ddof=1)) if n > 1 else 0.0
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    
    # Moda empírica
    series = pd.Series(arr)
    modes = series.mode()
    mode_val = float(modes.iloc[0]) if len(modes) > 0 else mean_val
    
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    
    return {
        'n': int(n),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': mean_val,
        'median': float(np.median(arr)),
        'mode': mode_val,
        'var': var_val,
        'std': std_val,
        'range': float(np.max(arr) - np.min(arr)),
        'iqr': float(p75 - p25),
        'cv': float(std_val / mean_val) if mean_val != 0 else None,
        'skewness': float(series.skew()) if n > 2 else 0.0,
        'kurtosis': float(series.kurtosis()) if n > 2 else 0.0,
        'percentiles': {f'p{p}': float(np.percentile(arr, p)) for p in percentiles}
    }


def frequency_table(data, max_categories=50):
    """
    Genera tabla de frecuencias absolutas, relativas y acumuladas para datos discretos.
    
    Args:
        data: Array de valores discretos.
        max_categories: Límite máximo de categorías a mostrar individualmente.
        
    Returns:
        Lista de diccionarios con value, frequency, relative_freq, percentage, cum_percentage.
    """
    clean = pd.Series(data).dropna()
    total_n = len(clean)
    if total_n == 0:
        return []
        
    val_counts = clean.value_counts().sort_index()
    
    table = []
    cum_freq = 0
    
    for val, freq in val_counts.items():
        cum_freq += freq
        rel_freq = freq / total_n
        table.append({
            'value': int(val) if float(val).is_integer() else round(float(val), 2),
            'frequency': int(freq),
            'relative_freq': round(float(rel_freq), 6),
            'percentage': round(float(rel_freq * 100), 2),
            'cum_percentage': round(float((cum_freq / total_n) * 100), 2)
        })
        
    return table


def histogram_data(data, n_bins=30):
    """
    Calcula los datos para un histograma continuo con Plotly.js.
    """
    arr = np.array(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {'bins': [], 'counts': []}
        
    counts, edges = np.histogram(arr, bins=n_bins)
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
    
    return {
        'bin_centers': [float(c) for c in centers],
        'bin_edges': [float(e) for e in edges],
        'counts': [int(c) for c in counts]
    }


def kde_data(data, n_points=150):
    """
    Calcula la estimación de densidad por kernel gaussiano (KDE) para gráficos continuos.
    """
    arr = np.array(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return {'x': [], 'y': []}
        
    try:
        kde = stats.gaussian_kde(arr)
        x_min = float(np.min(arr))
        x_max = float(np.max(arr))
        xs = np.linspace(x_min, x_max, n_points)
        ys = kde(xs)
        return {
            'x': [float(x) for x in xs],
            'y': [float(y) for y in ys]
        }
    except Exception:
        return {'x': [], 'y': []}


def ecdf_data(data, max_points=500):
    """
    Calcula los puntos para graficar la Función de Distribución Acumulada Empírica (ECDF).
    """
    arr = np.array(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {'x': [], 'y': []}
        
    sorted_data = np.sort(arr)
    n = len(sorted_data)
    y_vals = np.arange(1, n + 1) / n
    
    # Submuestrear si n es muy grande para aligerar la gráfica
    if n > max_points:
        indices = np.linspace(0, n - 1, max_points, dtype=int)
        sorted_data = sorted_data[indices]
        y_vals = y_vals[indices]
        
    return {
        'x': [float(x) for x in sorted_data],
        'y': [float(y) for y in y_vals]
    }
