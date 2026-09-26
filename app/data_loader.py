"""
Módulo de carga y exploración de datos para el dataset de CS:GO.
Detecta automáticamente los archivos CSV descargados por KaggleHub o ubicados en data/,
calcula estadísticas descriptivas, y sugiere variables discretas y continuas.
"""

import os
import glob
import pandas as pd
import numpy as np


def list_available_datasets():
    """
    Busca y lista todos los archivos CSV disponibles en la carpeta data/,
    en el directorio raíz y en el caché de KaggleHub.
    
    Returns:
        Lista de diccionarios con información de cada archivo CSV.
    """
    search_dirs = [
        os.path.join(os.path.dirname(__file__), '..', 'data'),
        os.path.join(os.path.dirname(__file__), '..'),
        os.path.join(os.path.expanduser('~'), '.cache', 'kagglehub', 'datasets'),
        os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'kagglehub', 'datasets'),
    ]
    
    files_found = {}
    
    descriptions = {
        'dataset.csv': 'Dataset principal seleccionado para el modelo (50,000 registros)',
        'mm_master_demos.csv': 'Partidas de Matchmaking oficial de Valve (Ranks 1-18, Daño, Tiempos, Equipamiento)',
        'esea_meta_demos.part1.csv': 'Metadatos de rondas ESEA (Ronda, Tiempos de inicio/fin, Economía)',
        'esea_meta_demos.part2.csv': 'Metadatos de rondas ESEA (Parte 2)',
        'esea_master_kills_demos.part1.csv': 'Eventos de bajas/kills (Arma, Jugadores vivos, Tiempos)',
        'esea_master_dmg_demos.part1.csv': 'Eventos de daño detallados ESEA (HP damage, Armaduras, Hitboxes)',
        'mm_grenades_demos.csv': 'Eventos de granadas y utilidad en Matchmaking',
    }
    
    for base_dir in search_dirs:
        if os.path.exists(base_dir):
            for root, _, filenames in os.walk(base_dir):
                for f in filenames:
                    if f.lower().endswith('.csv'):
                        full_path = os.path.abspath(os.path.join(root, f))
                        base_name = os.path.basename(full_path)
                        if base_name not in files_found:
                            size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 2)
                            files_found[base_name] = {
                                'filename': base_name,
                                'filepath': full_path,
                                'size_mb': size_mb,
                                'description': descriptions.get(base_name, 'Archivo CSV del dataset CS:GO')
                            }
                            
    # Ordenar dando prioridad a dataset.csv, mm_master_demos y esea_meta
    priority_order = ['dataset.csv', 'mm_master_demos.csv', 'esea_meta_demos.part1.csv', 'esea_meta_demos.part2.csv']
    result = []
    for prio in priority_order:
        if prio in files_found:
            result.append(files_found.pop(prio))
    for item in sorted(files_found.values(), key=lambda x: x['size_mb']):
        result.append(item)
        
    return result


def find_dataset_csv(data_dir=None, preferred_file=None):
    """
    Busca automáticamente el archivo CSV del dataset con prioridad inteligente.
    
    Args:
        data_dir: Directorio donde buscar (opcional).
        preferred_file: Nombre de archivo preferido (opcional).
        
    Returns:
        Ruta absoluta al archivo CSV encontrado.
    """
    available = list_available_datasets()
    if not available:
        raise FileNotFoundError(
            "No se encontró ningún archivo CSV del dataset de CS:GO. "
            "Por favor, descargue el dataset usando kagglehub.dataset_download('skihikingkevin/csgo-matchmaking-damage')."
        )
        
    if preferred_file:
        for item in available:
            if preferred_file.lower() in item['filename'].lower():
                return item['filepath']
                
    # Prioridad 1: dataset.csv
    for item in available:
        if item['filename'] == 'dataset.csv':
            return item['filepath']
            
    # Prioridad 2: mm_master_demos.csv
    for item in available:
        if 'mm_master_demos' in item['filename']:
            return item['filepath']
            
    # Prioridad 3: esea_meta_demos
    for item in available:
        if 'meta' in item['filename']:
            return item['filepath']
            
    return available[0]['filepath']


def load_dataset(filepath=None, nrows=100000, max_cols=50):
    """
    Carga el dataset CSV con pandas de manera optimizada y robusta.
    Añade transformaciones estadísticas pertinentes si están disponibles en las columnas
    (como 'duration' = end_seconds - start_seconds).
    
    Args:
        filepath: Ruta al archivo CSV. Si es None, busca automáticamente.
        nrows: Número máximo de filas a cargar para garantizar interactividad veloz.
        max_cols: Máximo de columnas a procesar.
        
    Returns:
        Tupla (DataFrame, diccionario de metadatos del dataset).
    """
    if filepath is None:
        filepath = find_dataset_csv()
        
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
    file_size_bytes = os.path.getsize(filepath)
    if file_size_bytes == 0:
        raise ValueError(f"El archivo está vacío: {filepath}")
        
    # Leer encabezados para conocer columnas disponibles
    header_df = pd.read_csv(filepath, nrows=0)
    all_columns = list(header_df.columns)
    
    # Determinar columnas a cargar (eliminar Unnamed si es solo índice)
    cols_to_use = [c for c in all_columns if not c.startswith('Unnamed:') or c == all_columns[0]][:max_cols]
    
    # Cargar datos con límite de filas para datasets masivos
    df = pd.read_csv(filepath, usecols=cols_to_use, nrows=nrows)
    
    # Limpiar columnas de índice no deseadas si tienen nombre 'Unnamed: 0'
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    # TRANSFORMAR O GENERAR VARIABLES DERIVADAS PERTINENTES:
    # Si existen start_seconds y end_seconds, calcular la duración continua de la ronda
    if 'start_seconds' in df.columns and 'end_seconds' in df.columns:
        if 'duration' not in df.columns:
            dur = df['end_seconds'] - df['start_seconds']
            # Filtrar valores inconsistentes de duración
            df['duration'] = dur.clip(lower=0.0)
            
    # Si existe att_rank en mm_master_demos, asegurarse de que 0 (unranked) se maneje adecuadamente
    if 'att_rank' in df.columns:
        # Los rangos oficiales de CS:GO van de 1 a 18
        df['att_rank'] = pd.to_numeric(df['att_rank'], errors='coerce')
        
    # Metadatos del dataset
    info = {
        'filepath': os.path.abspath(filepath),
        'filename': os.path.basename(filepath),
        'file_size_mb': round(file_size_bytes / (1024 * 1024), 2),
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'null_counts': {col: int(cnt) for col, cnt in df.isnull().sum().items()},
        'null_percentage': {col: round(cnt / len(df) * 100, 2) for col, cnt in df.isnull().sum().items()},
        'n_duplicates': int(df.duplicated().sum()),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }
    
    return df, info


def get_column_stats(df, max_cols=50):
    """
    Calcula estadísticas descriptivas detalladas para las columnas del DataFrame.
    
    Args:
        df: DataFrame de pandas.
        max_cols: Máximo de columnas a procesar.
        
    Returns:
        Lista de diccionarios con estadísticas por columna.
    """
    stats_list = []
    cols_to_process = list(df.columns)[:max_cols]
    
    for col in cols_to_process:
        s = df[col]
        n_null = int(s.isnull().sum())
        null_pct = round((n_null / len(df)) * 100, 2) if len(df) > 0 else 0.0
        n_unique = int(s.nunique(dropna=True))
        
        stat = {
            'name': col,
            'dtype': str(s.dtype),
            'n_null': n_null,
            'null_pct': null_pct,
            'n_unique': n_unique,
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
            'median': None,
            'skewness': None,
            'kurtosis': None
        }
        
        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            if len(clean) > 0:
                stat['min'] = float(np.min(clean))
                stat['max'] = float(np.max(clean))
                stat['mean'] = float(np.mean(clean))
                stat['std'] = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
                stat['median'] = float(np.median(clean))
                if len(clean) > 2:
                    stat['skewness'] = float(pd.Series(clean).skew())
                    stat['kurtosis'] = float(pd.Series(clean).kurtosis())
                    
        stats_list.append(stat)
        
    return stats_list


def suggest_discrete_columns(df):
    """
    Sugiere columnas candidatas para ser modeladas como Variable Aleatoria Discreta.
    
    Criterios académicos rigurosos:
    - No asume que un tipo entero es automáticamente discreto.
    - Analiza significado conceptual, cardinalidad finita (típicamente entre 2 y 50 valores únicos).
    - Excluye identificadores, timestamps, coordenadas espaciales.
    - Prioriza variables representativas de estados del juego (ej. att_rank, round, ct_alive).
    
    Args:
        df: DataFrame de pandas.
        
    Returns:
        Lista ordenada de nombres de columnas recomendadas.
    """
    candidates = []
    exclude_keywords = ['file', 'id', 'name', 'tick', 'date', 'pos_x', 'pos_y', 'index', 'unnamed', 'site']
    
    priority_map = {
        'att_rank': 100,       # Rango de habilidad del jugador (1 a 18): ideal para matchmaking
        'vic_rank': 90,        # Rango de la víctima
        'round': 85,           # Número de ronda (1 a 30)
        'ct_alive': 80,        # Jugadores vivos CT (0 a 5)
        't_alive': 80,         # Jugadores vivos T (0 a 5)
        'award': 70,           # Dinero de recompensa
        'hp_dmg': 65,          # Daño en HP (valores enteros)
        'winner_side': 60,     # Bando ganador (0/1 codificado)
    }
    
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in exclude_keywords):
            continue
            
        n_unique = df[col].nunique(dropna=True)
        is_num = pd.api.types.is_numeric_dtype(df[col])
        
        # Debe tener cardinalidad discreta finita (>= 2 y <= 60)
        if 2 <= n_unique <= 60:
            if is_num:
                # Comprobar si los valores son enteros o conteos
                non_null = df[col].dropna()
                if np.all(np.equal(np.mod(non_null, 1), 0)):
                    score = priority_map.get(col_lower, 50 - abs(n_unique - 18))
                    candidates.append((col, score))
            elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
                if n_unique <= 10:
                    candidates.append((col, 30))
                    
    # Ordenar por puntuación de idoneidad descendente
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in candidates]


def suggest_continuous_columns(df):
    """
    Sugiere columnas candidatas para ser modeladas como Variable Aleatoria Continua.
    
    Criterios académicos rigurosos:
    - Tipo numérico float o entero con alta cardinalidad (> 60 valores únicos).
    - Representa una magnitud física, temporal o económica medible en la escala real.
    - Excluye identificadores, timestamps continuos que son índices, coordenadas arbitrarias.
    - Prioriza variables como duración de ronda (duration), tiempo de evento (seconds), equipamiento (ct_eq_val).
    
    Args:
        df: DataFrame de pandas.
        
    Returns:
        Lista ordenada de nombres de columnas recomendadas.
    """
    candidates = []
    exclude_keywords = ['file', 'id', 'name', 'tick', 'date', 'pos_x', 'pos_y', 'index', 'unnamed']
    
    priority_map = {
        'duration': 100,         # Duración de ronda en segundos (end_seconds - start_seconds)
        'round_duration': 100,
        'seconds': 90,           # Segundo del evento en la ronda/partida
        'hp_dmg': 75,            # Daño continuo si se modela como continuo
        'ct_eq_val': 85,         # Valor económico del equipo CT ($)
        't_eq_val': 85,          # Valor económico del equipo T ($)
        'avg_match_rank': 80,    # Rango promedio del match (escala continua decimal)
    }
    
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in exclude_keywords):
            continue
            
        if pd.api.types.is_numeric_dtype(df[col]):
            n_unique = df[col].nunique(dropna=True)
            if n_unique > 50:
                score = priority_map.get(col_lower, 50)
                candidates.append((col, score))
                
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in candidates]


def download_kaggle_dataset():
    """
    Descarga el dataset de CS:GO desde Kaggle utilizando kagglehub y copia los CSV a data/.
    
    Returns:
        Ruta del directorio donde quedaron los archivos.
    """
    import kagglehub
    import shutil
    
    print("Descargando dataset de CS:GO vía KaggleHub...")
    path = kagglehub.dataset_download("skihikingkevin/csgo-matchmaking-damage")
    
    dest_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(dest_dir, exist_ok=True)
    
    copied = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith('.csv'):
                src = os.path.join(root, f)
                dst = os.path.join(dest_dir, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied.append(f)
                    
    return {
        'source_path': path,
        'dest_dir': dest_dir,
        'files_copied': copied
    }
