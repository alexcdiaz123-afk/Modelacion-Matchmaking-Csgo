"""
Módulo de preprocesamiento y limpieza de datos.
Todas las operaciones trabajan sobre una copia del DataFrame original.
"""

import pandas as pd
import numpy as np


def create_clean_copy(df):
    """
    Crea una copia limpia del DataFrame original.
    
    Args:
        df: DataFrame original.
    
    Returns:
        Copia del DataFrame.
    """
    return df.copy()


def remove_duplicates(df):
    """
    Elimina filas duplicadas del DataFrame.
    
    Args:
        df: DataFrame.
    
    Returns:
        Tupla con (DataFrame sin duplicados, número de filas eliminadas).
    """
    n_before = len(df)
    df_clean = df.drop_duplicates()
    n_removed = n_before - len(df_clean)
    return df_clean, n_removed


def handle_nulls(df, strategy='drop', fill_value=None):
    """
    Maneja valores nulos en el DataFrame.
    
    Args:
        df: DataFrame.
        strategy: Estrategia ('drop', 'mean', 'median', 'mode', 'constant').
        fill_value: Valor para rellenar cuando strategy='constant'.
    
    Returns:
        DataFrame con valores nulos manejados.
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        df_clean = df_clean.dropna()
    elif strategy == 'mean':
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif strategy == 'median':
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif strategy == 'mode':
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0)
    elif strategy == 'constant':
        df_clean = df_clean.fillna(fill_value if fill_value is not None else 0)
    
    return df_clean


def convert_types(df, column_types):
    """
    Convierte tipos de datos de columnas específicas.
    
    Args:
        df: DataFrame.
        column_types: Diccionario {nombre_columna: tipo_objetivo}.
    
    Returns:
        DataFrame con tipos convertidos.
    """
    df_clean = df.copy()
    for col, dtype in column_types.items():
        if col in df_clean.columns:
            try:
                if dtype == 'int':
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('Int64')
                elif dtype == 'float':
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                elif dtype == 'str':
                    df_clean[col] = df_clean[col].astype(str)
                elif dtype == 'category':
                    df_clean[col] = df_clean[col].astype('category')
                elif dtype == 'datetime':
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            except Exception:
                pass  # Mantener tipo original si falla la conversión
    return df_clean


def remove_invalid_values(df, column, min_val=None, max_val=None):
    """
    Elimina valores inválidos fuera de un rango.
    
    Args:
        df: DataFrame.
        column: Nombre de la columna.
        min_val: Valor mínimo permitido.
        max_val: Valor máximo permitido.
    
    Returns:
        DataFrame filtrado.
    """
    df_clean = df.copy()
    if column in df_clean.columns:
        if min_val is not None:
            df_clean = df_clean[df_clean[column] >= min_val]
        if max_val is not None:
            df_clean = df_clean[df_clean[column] <= max_val]
    return df_clean


def filter_outliers(df, column, method='iqr', threshold=1.5):
    """
    Filtra valores extremos (outliers).
    
    Args:
        df: DataFrame.
        column: Nombre de la columna.
        method: Método ('iqr' o 'zscore').
        threshold: Umbral para considerar outlier.
    
    Returns:
        DataFrame sin outliers.
    """
    df_clean = df.copy()
    if column not in df_clean.columns:
        return df_clean
    
    if method == 'iqr':
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        df_clean = df_clean[(df_clean[column] >= lower) & (df_clean[column] <= upper)]
    elif method == 'zscore':
        mean = df_clean[column].mean()
        std = df_clean[column].std()
        if std > 0:
            z_scores = (df_clean[column] - mean) / std
            df_clean = df_clean[z_scores.abs() <= threshold]
    
    return df_clean


def get_cleaning_summary(df_original, df_clean):
    """
    Genera un resumen comparativo antes y después de la limpieza.
    
    Args:
        df_original: DataFrame original.
        df_clean: DataFrame limpio.
    
    Returns:
        Diccionario con métricas comparativas.
    """
    return {
        'before': {
            'rows': len(df_original),
            'cols': len(df_original.columns),
            'nulls': int(df_original.isnull().sum().sum()),
            'duplicates': int(df_original.duplicated().sum()),
        },
        'after': {
            'rows': len(df_clean),
            'cols': len(df_clean.columns),
            'nulls': int(df_clean.isnull().sum().sum()),
            'duplicates': int(df_clean.duplicated().sum()),
        },
        'rows_removed': len(df_original) - len(df_clean),
        'nulls_removed': int(df_original.isnull().sum().sum()) - int(df_clean.isnull().sum().sum()),
    }
