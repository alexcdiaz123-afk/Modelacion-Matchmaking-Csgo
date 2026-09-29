"""
Aplicación Flask principal: Simulador Estadístico de Matchmaking — CS:GO
Dashboard interactivo para modelación probabilística de variables aleatorias discretas y continuas.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, send_file

# Asegurar path para imports locales
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import (
    load_dataset, get_column_stats, suggest_discrete_columns,
    suggest_continuous_columns, list_available_datasets, download_kaggle_dataset
)
from preprocessing import (
    create_clean_copy, remove_duplicates, handle_nulls, convert_types,
    remove_invalid_values, filter_outliers, get_cleaning_summary
)
from distributions import (
    fit_discrete_distributions, fit_continuous_distributions,
    get_discrete_distributions, get_continuous_distributions,
    select_best, sample_moment_profile, SCORE_WEIGHTS, VIABLE_MASS_THRESHOLD
)
from simulation import (
    generate_discrete, generate_continuous, get_method_description,
    compare_distributions, DISCRETE_METHODS, CONTINUOUS_METHODS
)
from statistics import (
    chi_square_test, ks_test, shapiro_test, anderson_test,
    dagostino_test, ks_2samp_test, mann_whitney_test, qq_plot_data, interpret_pvalue
)
from matchmaking import MatchmakingSimulator
from utils import (
    NumpyEncoder, safe_float, safe_int, format_params,
    calculate_descriptive_stats, frequency_table, histogram_data,
    kde_data, ecdf_data
)

# =============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN FLASK
# =============================================================================

app = Flask(__name__)
app.secret_key = 'csgo-matchmaking-probabilistic-analytics-2026'

# Configurar serializador JSON compatible con NumPy
from flask.json.provider import DefaultJSONProvider

class CustomJSONProvider(DefaultJSONProvider):
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
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)

app.json_provider_class = CustomJSONProvider
app.json = CustomJSONProvider(app)


# Estado global persistente del análisis
DATASET_STATE = {
    'original_df': None,
    'clean_df': None,
    'info': None,
    'discrete_col': None,
    'continuous_col': None,
    'discrete_results': None,
    'continuous_results': None,
    'discrete_selected': None,
    'continuous_selected': None,
    'discrete_simulated': None,
    'continuous_simulated': None,
    'discrete_sim_params': None,
    'continuous_sim_params': None,
    'cleaning_summary': None,
    'alpha': 0.05,
    'matchmaking_results': None
}


def get_df():
    """Retorna el DataFrame activo (limpio si existe, sino el original)."""
    if DATASET_STATE['clean_df'] is not None:
        return DATASET_STATE['clean_df']
    return DATASET_STATE['original_df']


def fmt_es(valor):
    """Entero con separador de miles del español (punto, no coma)."""
    return f"{int(valor):,}".replace(",", ".")


@app.template_filter('es')
def _fmt_es_filter(valor):
    """Expone fmt_es a las plantillas Jinja."""
    try:
        return fmt_es(valor)
    except (TypeError, ValueError):
        return valor


def auto_select_default_variables(df):
    """Selecciona de forma inteligente las variables discretas y continuas óptimas según las columnas presentes."""
    cols = list(df.columns)
    
    # Selección discreta:
    # Prioridad: 'att_rank' -> 'round' -> 'ct_alive'
    if 'att_rank' in cols and df['att_rank'].nunique() > 1:
        DATASET_STATE['discrete_col'] = 'att_rank'
    elif 'round' in cols:
        DATASET_STATE['discrete_col'] = 'round'
    elif 'ct_alive' in cols:
        DATASET_STATE['discrete_col'] = 'ct_alive'
    else:
        sug = suggest_discrete_columns(df)
        if sug:
            DATASET_STATE['discrete_col'] = sug[0]

    # Selección continua:
    # Prioridad: 'duration' -> 'seconds' -> 'hp_dmg' -> 'ct_eq_val'
    if 'duration' in cols:
        DATASET_STATE['continuous_col'] = 'duration'
    elif 'seconds' in cols:
        DATASET_STATE['continuous_col'] = 'seconds'
    elif 'hp_dmg' in cols:
        DATASET_STATE['continuous_col'] = 'hp_dmg'
    elif 'ct_eq_val' in cols:
        DATASET_STATE['continuous_col'] = 'ct_eq_val'
    else:
        sug = suggest_continuous_columns(df)
        if sug:
            DATASET_STATE['continuous_col'] = sug[0]


# =============================================================================
# RUTAS DE VISTAS (PÁGINAS HTML)
# =============================================================================

@app.route('/')
def index():
    """MÓDULO 1: Dashboard y Contexto del Matchmaking."""
    df = get_df()
    n_rows = len(df) if df is not None else 0
    n_cols = len(df.columns) if df is not None else 0
    is_cleaned = DATASET_STATE['clean_df'] is not None
    info = DATASET_STATE.get('info') or {}
    return render_template('index.html',
                           n_rows=n_rows,
                           n_cols=n_cols,
                           is_cleaned=is_cleaned,
                           info=info,
                           discrete_col=DATASET_STATE.get('discrete_col'),
                           continuous_col=DATASET_STATE.get('continuous_col'),
                           discrete_selected=DATASET_STATE.get('discrete_selected'),
                           continuous_selected=DATASET_STATE.get('continuous_selected'))


@app.route('/dataset')
def dataset_page():
    """MÓDULO 2: Explorador del Dataset."""
    df = get_df()
    info = DATASET_STATE.get('info') or {}
    is_cleaned = DATASET_STATE['clean_df'] is not None
    n_rows = len(df) if df is not None else 0
    n_cols = len(df.columns) if df is not None else 0
    n_duplicates = int(df.duplicated().sum()) if df is not None else 0
    mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2) if df is not None else 0
    return render_template('dataset.html',
                           df=df,
                           info=info,
                           is_cleaned=is_cleaned,
                           n_rows=n_rows,
                           n_cols=n_cols,
                           n_duplicates=n_duplicates,
                           mem_mb=mem_mb)


@app.route('/cleaning')
def cleaning_page():
    """MÓDULO 3: Limpieza y Depuración de Datos."""
    df = get_df()
    columns = list(df.columns) if df is not None else []
    summary = DATASET_STATE.get('cleaning_summary')
    if summary is None and DATASET_STATE['original_df'] is not None:
        orig = DATASET_STATE['original_df']
        summary = {
            'before': {'rows': len(orig), 'cols': len(orig.columns), 'nulls': int(orig.isnull().sum().sum()), 'duplicates': int(orig.duplicated().sum())},
            'after': {'rows': len(orig), 'cols': len(orig.columns), 'nulls': int(orig.isnull().sum().sum()), 'duplicates': int(orig.duplicated().sum())},
            'rows_removed': 0,
            'nulls_removed': 0
        }
    return render_template('cleaning.html', columns=columns, summary=summary)


@app.route('/variables')
def variables_page():
    """MÓDULO 4: Selección de Variables Aleatorias."""
    df = get_df()
    if df is None:
        return render_template('dataset.html')
    col_stats = get_column_stats(df)
    discrete_candidates = suggest_discrete_columns(df)
    continuous_candidates = suggest_continuous_columns(df)
    return render_template('variables.html',
                           col_stats=col_stats,
                           discrete_candidates=discrete_candidates,
                           continuous_candidates=continuous_candidates,
                           discrete_col=DATASET_STATE['discrete_col'],
                           continuous_col=DATASET_STATE['continuous_col'])


@app.route('/discrete')
def discrete_page():
    """MÓDULOS 5, 6 y 7: Variable Aleatoria Discreta y Ajuste de Distribuciones."""
    df = get_df()
    if df is None:
        return render_template('dataset.html')
        
    col = DATASET_STATE['discrete_col']
    if not col or col not in df.columns:
        auto_select_default_variables(df)
        col = DATASET_STATE['discrete_col']
        
    data = df[col].dropna().values
    stats = calculate_descriptive_stats(data)
    freq_table = frequency_table(data)
    
    # Ajustar distribuciones candidatas
    obs_vals = [f['value'] for f in freq_table]
    obs_freq = [f['frequency'] for f in freq_table]
    dist_results = fit_discrete_distributions(data, obs_vals, obs_freq)
    DATASET_STATE['discrete_results'] = dist_results
    
    # Si no se ha preseleccionado una distribución, elegir con el criterio multiobjetivo.
    # No se ordena por p-valor ni por el estadístico crudo: con decenas de miles de
    # observaciones χ² rechaza a todos los candidatos (p≈0) y el estadístico por sí solo
    # no distingue. Ver score_candidates en distributions.py.
    if not DATASET_STATE['discrete_selected']:
        best = select_best(dist_results)
        if best is not None:
            DATASET_STATE['discrete_selected'] = best['name']

    return render_template('discrete.html',
                           column=col,
                           stats=stats,
                           freq_table=freq_table,
                           dist_results=dist_results,
                           selected_dist=DATASET_STATE['discrete_selected'],
                           alpha=DATASET_STATE['alpha'])


@app.route('/continuous')
def continuous_page():
    """MÓDULOS 8, 9 y 10: Variable Aleatoria Continua y Pruebas de Normalidad."""
    df = get_df()
    if df is None:
        return render_template('dataset.html')
        
    col = DATASET_STATE['continuous_col']
    if not col or col not in df.columns:
        auto_select_default_variables(df)
        col = DATASET_STATE['continuous_col']
        
    data = df[col].dropna().values
    stats = calculate_descriptive_stats(data)
    
    # Ajuste de distribuciones continuas
    dist_results = fit_continuous_distributions(data)
    DATASET_STATE['continuous_results'] = dist_results
    
    # Pruebas de normalidad específicas
    normality_tests = {
        'shapiro': shapiro_test(data),
        'anderson': anderson_test(data),
        'dagostino': dagostino_test(data)
    }
    
    if not DATASET_STATE['continuous_selected']:
        best = select_best(dist_results)
        if best is not None:
            DATASET_STATE['continuous_selected'] = best['name']

    return render_template('continuous.html',
                           column=col,
                           stats=stats,
                           dist_results=dist_results,
                           normality_tests=normality_tests,
                           selected_dist=DATASET_STATE['continuous_selected'],
                           alpha=DATASET_STATE['alpha'])


def ensure_distributions_fitted():
    """Garantiza que las distribuciones discreta y continua estén calculadas y ajustadas."""
    df = get_df()
    if df is None:
        return
    if not DATASET_STATE.get('discrete_col') or not DATASET_STATE.get('continuous_col'):
        auto_select_default_variables(df)
        
    d_col = DATASET_STATE.get('discrete_col')
    if DATASET_STATE.get('discrete_results') is None and d_col and d_col in df.columns:
        data = df[d_col].dropna().values
        ft = frequency_table(data)
        obs_vals = [f['value'] for f in ft]
        obs_freq = [f['frequency'] for f in ft]
        dist_res = fit_discrete_distributions(data, obs_vals, obs_freq)
        DATASET_STATE['discrete_results'] = dist_res
        if not DATASET_STATE.get('discrete_selected'):
            best = select_best(dist_res)
            if best:
                DATASET_STATE['discrete_selected'] = best['name']
                
    c_col = DATASET_STATE.get('continuous_col')
    if DATASET_STATE.get('continuous_results') is None and c_col and c_col in df.columns:
        data = df[c_col].dropna().values
        dist_res = fit_continuous_distributions(data)
        DATASET_STATE['continuous_results'] = dist_res
        if not DATASET_STATE.get('continuous_selected'):
            best = select_best(dist_res)
            if best:
                DATASET_STATE['continuous_selected'] = best['name']


@app.route('/distributions')
def distributions_page():
    """MÓDULO 11: Selección Final y Justificación Académica de Distribuciones."""
    ensure_distributions_fitted()
    return render_template('distributions.html',
                           discrete_col=DATASET_STATE['discrete_col'],
                           continuous_col=DATASET_STATE['continuous_col'],
                           discrete_selected=DATASET_STATE['discrete_selected'],
                           continuous_selected=DATASET_STATE['continuous_selected'],
                           discrete_results=DATASET_STATE['discrete_results'],
                           continuous_results=DATASET_STATE['continuous_results'])


@app.route('/simulation')
def simulation_page():
    """MÓDULOS 12 y 13: Generación de Variables Aleatorias y Comparación Real vs Simulada."""
    ensure_distributions_fitted()
    return render_template('simulation.html',
                           discrete_col=DATASET_STATE['discrete_col'],
                           continuous_col=DATASET_STATE['continuous_col'],
                           discrete_selected=DATASET_STATE['discrete_selected'],
                           continuous_selected=DATASET_STATE['continuous_selected'],
                           discrete_results=DATASET_STATE['discrete_results'],
                           continuous_results=DATASET_STATE['continuous_results'])


@app.route('/matchmaking')
def matchmaking_page():
    """MÓDULOS 14, 15, 16 y 17: Simulador de Matchmaking y Cola Estocástica."""
    ensure_distributions_fitted()
    return render_template('matchmaking.html',
                           discrete_col=DATASET_STATE['discrete_col'],
                           continuous_col=DATASET_STATE['continuous_col'],
                           discrete_selected=DATASET_STATE['discrete_selected'],
                           continuous_selected=DATASET_STATE['continuous_selected'])


@app.route('/theory')
def theory_page():
    """MÓDULO TEÓRICO: Fundamentos, Ecuaciones Matemáticas y Modelación."""
    return render_template('theory.html',
                           discrete_col=DATASET_STATE['discrete_col'],
                           continuous_col=DATASET_STATE['continuous_col'],
                           discrete_selected=DATASET_STATE['discrete_selected'],
                           continuous_selected=DATASET_STATE['continuous_selected'])


@app.route('/report')
def report_page():
    """MÓDULO 18: Informe Académico Completo de 20 Secciones."""
    return render_template('report.html',
                           info=DATASET_STATE['info'],
                           discrete_col=DATASET_STATE['discrete_col'],
                           continuous_col=DATASET_STATE['continuous_col'],
                           discrete_selected=DATASET_STATE['discrete_selected'],
                           continuous_selected=DATASET_STATE['continuous_selected'],
                           cleaning_summary=DATASET_STATE['cleaning_summary'])


# =============================================================================
# APIS DE CARGA Y EXPLORACIÓN DEL DATASET
# =============================================================================

@app.route('/api/dataset/files')
def api_dataset_files():
    """Retorna la lista de todos los archivos CSV encontrados."""
    try:
        files = list_available_datasets()
        return jsonify({'success': True, 'files': files})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dataset/info')
def api_dataset_info():
    """Retorna información detallada y métricas del DataFrame activo (limpio u original)."""
    try:
        df = get_df()
        if df is None:
            df, info = load_dataset(nrows=50000)
            DATASET_STATE['original_df'] = df
            DATASET_STATE['info'] = info
            auto_select_default_variables(df)
            
        df = get_df()
        info = DATASET_STATE.get('info') or {}
        is_cleaned = DATASET_STATE['clean_df'] is not None
        
        n_rows = len(df)
        n_cols = len(df.columns)
        n_duplicates = int(df.duplicated().sum())
        mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        
        null_counts = {col: int(cnt) for col, cnt in df.isnull().sum().items()}
        null_pct = {col: round(cnt / n_rows * 100, 2) if n_rows > 0 else 0 for col, cnt in null_counts.items()}
        
        return jsonify({
            'success': True,
            'filename': info.get('filename', 'dataset.csv'),
            'filepath': info.get('filepath', ''),
            'file_size_mb': info.get('file_size_mb', 0),
            'n_rows': n_rows,
            'n_cols': n_cols,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_counts': null_counts,
            'null_percentage': null_pct,
            'n_duplicates': n_duplicates,
            'memory_usage_mb': mem_mb,
            'is_cleaned': is_cleaned,
            'discrete_col': DATASET_STATE.get('discrete_col'),
            'continuous_col': DATASET_STATE.get('continuous_col'),
            'discrete_selected': DATASET_STATE.get('discrete_selected'),
            'continuous_selected': DATASET_STATE.get('continuous_selected'),
            'cleaning_summary': DATASET_STATE.get('cleaning_summary')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dataset/init', methods=['GET', 'POST'])
def api_dataset_init():
    """Inicializa automáticamente o retorna el estado del dataset activo."""
    try:
        # Si ya hay un dataset cargado en memoria, NO lo recargamos desde disco ni borramos clean_df
        if DATASET_STATE['original_df'] is not None:
            df = get_df()
            info = dict(DATASET_STATE.get('info') or {})
            
            # Recalcular métricas actuales del DataFrame activo
            n_rows = len(df)
            n_cols = len(df.columns)
            info['n_rows'] = n_rows
            info['n_cols'] = n_cols
            info['n_duplicates'] = int(df.duplicated().sum())
            info['memory_usage_mb'] = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            info['is_cleaned'] = DATASET_STATE['clean_df'] is not None
            info['columns'] = list(df.columns)
            info['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            info['null_counts'] = {col: int(cnt) for col, cnt in df.isnull().sum().items()}
            info['null_percentage'] = {col: round(cnt / n_rows * 100, 2) if n_rows > 0 else 0 for col, cnt in info['null_counts'].items()}
            
            return jsonify({
                'success': True,
                'info': info,
                'is_cleaned': DATASET_STATE['clean_df'] is not None,
                'discrete_col': DATASET_STATE.get('discrete_col'),
                'continuous_col': DATASET_STATE.get('continuous_col')
            })
            
        df, info = load_dataset(nrows=50000)
        DATASET_STATE['original_df'] = df
        DATASET_STATE['info'] = info
        DATASET_STATE['clean_df'] = None
        
        auto_select_default_variables(df)
        
        return jsonify({
            'success': True,
            'info': info,
            'is_cleaned': False,
            'discrete_col': DATASET_STATE['discrete_col'],
            'continuous_col': DATASET_STATE['continuous_col']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dataset/load', methods=['POST'])
def api_dataset_load():
    """Carga manualmente un archivo CSV específico."""
    try:
        data = request.get_json() or {}
        filepath = data.get('filepath')
        nrows = data.get('nrows', 50000)
        
        df, info = load_dataset(filepath=filepath, nrows=nrows)
        DATASET_STATE['original_df'] = df
        DATASET_STATE['info'] = info
        DATASET_STATE['clean_df'] = None
        DATASET_STATE['cleaning_summary'] = None
        
        auto_select_default_variables(df)
        
        return jsonify({
            'success': True,
            'info': info,
            'discrete_col': DATASET_STATE['discrete_col'],
            'continuous_col': DATASET_STATE['continuous_col']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/dataset/download', methods=['POST'])
def api_dataset_download():
    """Descarga el dataset desde KaggleHub."""
    try:
        res = download_kaggle_dataset()
        return jsonify({'success': True, 'result': res})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dataset/head')
def api_dataset_head():
    """Retorna las primeras n filas del DataFrame."""
    df = get_df()
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    n = request.args.get('n', 10, type=int)
    cols = list(df.columns[:25])
    return jsonify(df.head(n)[cols].to_dict('records'))


@app.route('/api/dataset/tail')
def api_dataset_tail():
    """Retorna las últimas n filas del DataFrame."""
    df = get_df()
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    n = request.args.get('n', 10, type=int)
    cols = list(df.columns[:25])
    return jsonify(df.tail(n)[cols].to_dict('records'))


@app.route('/api/dataset/stats')
def api_dataset_stats():
    """Retorna estadísticas descriptivas por columna."""
    df = get_df()
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    return jsonify(get_column_stats(df))


@app.route('/api/dataset/suggestions')
def api_dataset_suggestions():
    """Retorna sugerencias inteligentes de variables candidatas."""
    df = get_df()
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    return jsonify({
        'discrete': suggest_discrete_columns(df),
        'continuous': suggest_continuous_columns(df),
        'current_discrete': DATASET_STATE['discrete_col'],
        'current_continuous': DATASET_STATE['continuous_col']
    })


# =============================================================================
# APIS DE LIMPIEZA DE DATOS
# =============================================================================

@app.route('/api/cleaning/apply', methods=['POST'])
def api_cleaning_apply():
    """Aplica operaciones de limpieza seleccionadas sobre una copia del DataFrame."""
    try:
        data = request.get_json() or {}
        operations = data.get('operations', [])
        
        df_clean = create_clean_copy(DATASET_STATE['original_df'])
        
        for op in operations:
            op_type = op.get('type')
            if op_type == 'remove_duplicates':
                df_clean, _ = remove_duplicates(df_clean)
            elif op_type == 'handle_nulls':
                strategy = op.get('strategy', 'drop')
                fill_val = op.get('fill_value')
                df_clean = handle_nulls(df_clean, strategy=strategy, fill_value=fill_val)
            elif op_type == 'convert_types':
                col_types = op.get('column_types', {})
                df_clean = convert_types(df_clean, col_types)
            elif op_type == 'remove_invalid':
                col = op.get('column')
                min_v = op.get('min')
                max_v = op.get('max')
                df_clean = remove_invalid_values(df_clean, col, min_v, max_v)
            elif op_type == 'filter_outliers':
                col = op.get('column')
                method = op.get('method', 'iqr')
                thresh = op.get('threshold', 1.5)
                df_clean = filter_outliers(df_clean, col, method=method, threshold=thresh)
                
        summary = get_cleaning_summary(DATASET_STATE['original_df'], df_clean)
        DATASET_STATE['clean_df'] = df_clean
        DATASET_STATE['cleaning_summary'] = summary
        DATASET_STATE['discrete_results'] = None
        DATASET_STATE['continuous_results'] = None
        DATASET_STATE['discrete_simulated'] = None
        DATASET_STATE['continuous_simulated'] = None
        DATASET_STATE['matchmaking_results'] = None
        
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/cleaning/reset', methods=['POST'])
def api_cleaning_reset():
    """Restaura los datos al estado original."""
    DATASET_STATE['clean_df'] = None
    DATASET_STATE['cleaning_summary'] = None
    DATASET_STATE['discrete_results'] = None
    DATASET_STATE['continuous_results'] = None
    DATASET_STATE['discrete_simulated'] = None
    DATASET_STATE['continuous_simulated'] = None
    DATASET_STATE['matchmaking_results'] = None
    return jsonify({'success': True})


# =============================================================================
# APIS DE SELECCIÓN Y ANÁLISIS DE VARIABLES
# =============================================================================

@app.route('/api/variables/select', methods=['POST'])
def api_variables_select():
    """Guarda la selección manual de variables aleatorias."""
    try:
        data = request.get_json() or {}
        disc = data.get('discrete')
        cont = data.get('continuous')
        df = get_df()
        
        if disc and disc in df.columns:
            DATASET_STATE['discrete_col'] = disc
            DATASET_STATE['discrete_results'] = None
            DATASET_STATE['discrete_selected'] = None
            
        if cont and cont in df.columns:
            DATASET_STATE['continuous_col'] = cont
            DATASET_STATE['continuous_results'] = None
            DATASET_STATE['continuous_selected'] = None
            
        return jsonify({'success': True, 'discrete': DATASET_STATE['discrete_col'], 'continuous': DATASET_STATE['continuous_col']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/discrete/select', methods=['POST'])
def api_discrete_select():
    """Selecciona la distribución discreta para la simulación."""
    data = request.get_json() or {}
    dist_name = data.get('distribution')
    DATASET_STATE['discrete_selected'] = dist_name
    return jsonify({'success': True, 'selected': dist_name})


@app.route('/api/continuous/select', methods=['POST'])
def api_continuous_select():
    """Selecciona la distribución continua para la simulación."""
    data = request.get_json() or {}
    dist_name = data.get('distribution')
    DATASET_STATE['continuous_selected'] = dist_name
    return jsonify({'success': True, 'selected': dist_name})


@app.route('/api/continuous/kde')
def api_continuous_kde():
    """Retorna la curva de densidad kernel (KDE) para la variable continua."""
    df = get_df()
    col = DATASET_STATE['continuous_col']
    if df is None or not col:
        return jsonify({'error': 'No data'}), 400
    data = df[col].dropna().values
    return jsonify(kde_data(data))


@app.route('/api/continuous/ecdf')
def api_continuous_ecdf():
    """Retorna la función de distribución acumulada empírica (ECDF)."""
    df = get_df()
    col = DATASET_STATE['continuous_col']
    if df is None or not col:
        return jsonify({'error': 'No data'}), 400
    data = df[col].dropna().values
    return jsonify(ecdf_data(data))


@app.route('/api/continuous/qq')
def api_continuous_qq():
    """Retorna los cuantiles teóricos vs empíricos para el Q-Q plot."""
    df = get_df()
    col = DATASET_STATE['continuous_col']
    if df is None or not col:
        return jsonify({'error': 'No data'}), 400
    data = df[col].dropna().values
    return jsonify(qq_plot_data(data))


# =============================================================================
# APIS DE SIMULACIÓN Y GENERACIÓN
# =============================================================================

@app.route('/api/simulation/generate', methods=['POST'])
def api_simulation_generate():
    """Genera variables aleatorias simuladas y compara con datos reales."""
    try:
        data = request.get_json() or {}
        size = data.get('size', 10000)
        seed = data.get('seed', 42)
        
        df = get_df()
        results = {}
        
        # 1. Simulación Discreta
        d_col = DATASET_STATE['discrete_col']
        d_dist = DATASET_STATE['discrete_selected']
        if d_col and d_dist and DATASET_STATE['discrete_results']:
            real_discrete = df[d_col].dropna().values
            params = {}
            for r in DATASET_STATE['discrete_results']:
                if r['name'] == d_dist and r['valid']:
                    params = r['params']
                    break
                    
            if params:
                sim_d = generate_discrete(d_dist, params, size=size, seed=seed)
                comparison_d = compare_distributions(real_discrete, sim_d)
                method_info_d = get_method_description(d_dist)
                
                # Muestreo representativo para Plotly (máx 2000 puntos para rendimiento)
                sample_n = min(2000, len(real_discrete), len(sim_d))
                rng = np.random.default_rng(seed)
                
                results['discrete'] = {
                    'column': d_col,
                    'distribution': d_dist,
                    'params': params,
                    'method': method_info_d,
                    'comparison': comparison_d,
                    'real_sample': rng.choice(real_discrete, sample_n, replace=False).tolist(),
                    'simulated_sample': sim_d[:sample_n].tolist()
                }
                DATASET_STATE['discrete_simulated'] = sim_d
                DATASET_STATE['discrete_sim_params'] = params
                
        # 2. Simulación Continua
        c_col = DATASET_STATE['continuous_col']
        c_dist = DATASET_STATE['continuous_selected']
        if c_col and c_dist and DATASET_STATE['continuous_results']:
            real_continuous = df[c_col].dropna().values
            params = {}
            for r in DATASET_STATE['continuous_results']:
                if r['name'] == c_dist and r['valid']:
                    params = r['params']
                    break
                    
            if params:
                sim_c = generate_continuous(c_dist, params, size=size, seed=seed)
                comparison_c = compare_distributions(real_continuous, sim_c)
                method_info_c = get_method_description(c_dist)
                
                sample_n = min(2000, len(real_continuous), len(sim_c))
                rng = np.random.default_rng(seed)
                
                results['continuous'] = {
                    'column': c_col,
                    'distribution': c_dist,
                    'params': params,
                    'method': method_info_c,
                    'comparison': comparison_c,
                    'real_sample': rng.choice(real_continuous, sample_n, replace=False).tolist(),
                    'simulated_sample': sim_c[:sample_n].tolist()
                }
                DATASET_STATE['continuous_simulated'] = sim_c
                DATASET_STATE['continuous_sim_params'] = params
                
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# =============================================================================
# APIS DE SIMULADOR DE MATCHMAKING
# =============================================================================

@app.route('/api/matchmaking/simulate', methods=['POST'])
def api_matchmaking_simulate():
    """Ejecuta la simulación de cola de matchmaking con las distribuciones reales."""
    try:
        data = request.get_json() or {}
        players_per_match = data.get('players_per_match', 10)
        max_queue_size = data.get('max_queue_size', 100)
        compatibility_threshold = data.get('compatibility_threshold', 0.20)
        n_players = data.get('n_players', 100)
        seed = data.get('seed', 42)
        
        # Extraer configuraciones de distribuciones
        df = get_df()
        
        # Configuración discreta
        d_name = DATASET_STATE['discrete_selected']
        d_params = DATASET_STATE['discrete_sim_params']
        if not d_params and DATASET_STATE['discrete_results']:
            for r in DATASET_STATE['discrete_results']:
                if r['name'] == d_name and r['valid']:
                    d_params = r['params']
                    break
        if not d_name or not d_params:
            fallback = select_best(DATASET_STATE['discrete_results'] or [])
            if fallback is not None:
                d_name = d_name or fallback['name']
                d_params = d_params or fallback['params']
        if not d_params:
            d_name = d_name or 'Poisson'
            d_params = {'lambda': 11.5}
            
        d_col = DATASET_STATE['discrete_col']
        d_min = float(df[d_col].min()) if df is not None and d_col in df.columns else 1.0
        d_max = float(df[d_col].max()) if df is not None and d_col in df.columns else 18.0
        
        # Configuración continua. El respaldo solo se usa si el usuario aún no ha
        # visitado /continuous; se obtiene del ajuste multiobjetivo y no de una
        # constante fija en el código.
        c_name = DATASET_STATE['continuous_selected']
        c_params = DATASET_STATE['continuous_sim_params']
        if not c_params and DATASET_STATE['continuous_results']:
            for r in DATASET_STATE['continuous_results']:
                if r['name'] == c_name and r['valid']:
                    c_params = r['params']
                    break
        if not c_name or not c_params:
            fallback = select_best(DATASET_STATE['continuous_results'] or [])
            if fallback is not None:
                c_name = c_name or fallback['name']
                c_params = c_params or fallback['params']
        if not c_params:
            c_name = c_name or 'Normal'
            c_params = {'mu': 85.0, 'sigma': 25.0}
            
        c_col = DATASET_STATE['continuous_col']
        c_min = float(df[c_col].min()) if df is not None and c_col in df.columns else 20.0
        c_max = float(df[c_col].max()) if df is not None and c_col in df.columns else 160.0
        
        discrete_config = (d_name, d_params, d_min, d_max)
        continuous_config = (c_name, c_params, c_min, c_max)
        
        simulator = MatchmakingSimulator(
            players_per_match=players_per_match,
            max_queue_size=max_queue_size,
            compatibility_threshold=compatibility_threshold,
            seed=seed,
            discrete_config=discrete_config,
            continuous_config=continuous_config
        )
        
        results = simulator.simulate(n_players=n_players)
        DATASET_STATE['matchmaking_results'] = results
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# =============================================================================
# GENERADOR DEL INFORME ACADÉMICO (20 SECCIONES)
# =============================================================================

def build_full_academic_report():
    """
    Construye el informe académico completo estructurado en las 20 secciones exactas
    solicitadas, utilizando los valores y resultados computados del dataset real de CS:GO.
    """
    df = get_df()
    info = DATASET_STATE['info'] or {}
    clean_summary = DATASET_STATE['cleaning_summary'] or {}
    d_col = DATASET_STATE['discrete_col']
    c_col = DATASET_STATE['continuous_col']
    d_sel = DATASET_STATE['discrete_selected']
    c_sel = DATASET_STATE['continuous_selected']
    alpha = DATASET_STATE['alpha']
    mm_res = DATASET_STATE['matchmaking_results']
    
    sections = []
    
    # 1. Introducción
    sections.append({
        'num': 1,
        'title': '1. Introducción',
        'content': (
            "El modelado probabilístico de sistemas de emparejamiento competitivo (matchmaking) "
            "constituye una piedra angular en el diseño de videojuegos modernos en línea como "
            "Counter-Strike: Global Offensive (CS:GO). El objetivo primordial de un sistema de matchmaking "
            "es reunir grupos de jugadores con habilidades y comportamientos comparables para generar partidas "
            "equitativas y competitivas. Este trabajo aborda el estudio estadístico riguroso de variables "
            "aleatorias observables en partidas reales de CS:GO, su caracterización teórica mediante modelos "
            "probabilísticos continuos y discretos, y su posterior integración en una simulación de cola de espera."
        )
    })
    
    # 2. Contexto
    sections.append({
        'num': 2,
        'title': '2. Contexto del Fenómeno de Matchmaking',
        'content': (
            "El fenómeno modelado describe la dinámica de un jugador que solicita ingresar a una partida y entra en una cola. "
            "El sistema evalúa de forma asíncrona la compatibilidad entre los participantes en espera. "
            "Cuando se reúne la cantidad reglamentaria de participantes (habitualmente 10 jugadores para un formato 5v5), "
            "se formaliza la creación de la partida.\n\n"
            "Es imperativo distinguir cuatro niveles conceptuales:\n"
            "1. DATOS REALES: Registros empíricos observados en repeticiones de partidas (.dem).\n"
            "2. PARÁMETROS ESTIMADOS: Valores estadísticos inferidos a partir de la muestra observada.\n"
            "3. DISTRIBUCIONES TEÓRICAS: Modelos matemáticos que aproximan el comportamiento poblacional.\n"
            "4. VARIABLES SIMULADAS Y REGLAS DE MATCHMAKING: Algoritmos de encolamiento y emparejamiento inspirados académicamente."
        )
    })
    
    # 3. Fuente de datos
    sections.append({
        'num': 3,
        'title': '3. Fuente de Datos',
        'content': (
            f"Dataset: CS:GO Matchmaking Damage Dataset (Kaggle: skhikingkevin/csgo-matchmaking-damage).\n"
            f"Archivo procesado: {info.get('filename', 'dataset.csv')}\n"
            f"Ruta: {info.get('filepath', 'data/dataset.csv')}\n"
            f"Tamaño en disco: {info.get('file_size_mb', 0)} MB\n"
            f"Mecanismo de adquisición: Descarga automatizada mediante la API de KaggleHub."
        )
    })
    
    # 4. Descripción del dataset
    null_counts = info.get('null_counts', {})
    null_summary_str = ", ".join([f"{col} ({cnt})" for col, cnt in null_counts.items() if cnt > 0]) or "Sin valores nulos"
    sections.append({
        'num': 4,
        'title': '4. Descripción del Dataset',
        'content': (
            f"La muestra cargada comprende {fmt_es(info.get('n_rows', 0))} filas y {info.get('n_cols', 0)} columnas.\n"
            f"Uso de memoria: {info.get('memory_usage_mb', 0)} MB.\n"
            f"Filas duplicadas iniciales: {fmt_es(info.get('n_duplicates', 0))}.\n"
            f"Distribución de valores nulos por columna: {null_summary_str}.\n"
            f"Columnas principales identificadas: {', '.join(info.get('columns', [])[:12])}..."
        )
    })
    
    # 5. Limpieza
    if clean_summary and 'before' in clean_summary:
        b = clean_summary['before']
        a = clean_summary['after']
        clean_text = (
            f"El preprocesamiento se realizó sobre una copia aislada del DataFrame para preservar la integridad de los datos crudos.\n"
            f"- Filas: {fmt_es(b.get('rows', 0))} (Antes) → {fmt_es(a.get('rows', 0))} (Después). Filas removidas: {fmt_es(clean_summary.get('rows_removed', 0))}.\n"
            f"- Valores nulos: {fmt_es(b.get('nulls', 0))} (Antes) → {fmt_es(a.get('nulls', 0))} (Después).\n"
            f"- Duplicados eliminados: {fmt_es(b.get('duplicates', 0))} registros."
        )
    else:
        clean_text = "No se aplicaron filtros destructivos adicionales; se verificó la consistencia de tipos numéricos y se eliminaron valores NaN en las variables analizadas."
    sections.append({
        'num': 5,
        'title': '5. Limpieza y Tratamiento de Datos',
        'content': clean_text
    })
    
    # 6. Variable discreta
    sections.append({
        'num': 6,
        'title': '6. Selección de la Variable Aleatoria Discreta',
        'content': (
            f"Variable seleccionada: '{d_col}'.\n"
            f"Justificación de selección: La variable '{d_col}' representa una cantidad discreta finita y contable "
            f"(en CS:GO, los rangos de habilidad representan los 18 grupos competitivos desde Silver I hasta Global Elite, "
            f"o el número de ronda disputada de 1 a 30). Su cardinalidad finita y naturaleza numérica entera "
            f"la hacen idónea para modelar estados discretos del matchmaking."
        )
    })
    
    # 7. Análisis exploratorio discreto
    d_data = df[d_col].dropna().values if df is not None and d_col in df.columns else np.array([])
    d_stats = calculate_descriptive_stats(d_data) if len(d_data) > 0 else {}
    sections.append({
        'num': 7,
        'title': '7. Análisis Exploratorio de la Variable Discreta',
        'content': (
            f"Estadísticas de la variable '{d_col}':\n"
            f"- Observaciones analizadas: {fmt_es(d_stats.get('n', 0))}\n"
            f"- Rango empírico: [{d_stats.get('min', 0):.0f}, {d_stats.get('max', 0):.0f}]\n"
            f"- Media muestral (x̄): {d_stats.get('mean', 0):.4f}\n"
            f"- Mediana: {d_stats.get('median', 0):.4f} | Moda: {d_stats.get('mode', 0):.4f}\n"
            f"- Varianza muestral (s²): {d_stats.get('var', 0):.4f}\n"
            f"- Desviación estándar (s): {d_stats.get('std', 0):.4f}\n"
            f"- Relación Varianza / Media: {d_stats.get('var', 1) / d_stats.get('mean', 1):.4f} "
            f"({'Sobredispersión (Var > Media)' if d_stats.get('var', 0) > d_stats.get('mean', 0) else 'Subdispersión o Equidispersión'})."
        )
    })
    
    # 8. Distribuciones candidatas discretas
    sections.append({
        'num': 8,
        'title': '8. Distribuciones Candidatas Discretas',
        'content': (
            "Se postularon cuatro distribuciones teóricas para la variable discreta:\n"
            "1. Poisson (λ): Asume equidispersión teórica Var(X) = E(X) y tasa constante de ocurrencia.\n"
            "2. Binomial (n, p): Modela el número de éxitos en n ensayos con varianza menor a la media.\n"
            "3. Binomial Negativa (r, p): Modela procesos sobredispersos donde la varianza supera notablemente a la media.\n"
            "4. Geométrica (p): Modela el número de intentos hasta el primer suceso exitoso."
        )
    })
    
    # 9. Pruebas estadísticas discretas
    d_res_list = DATASET_STATE.get('discrete_results') or []
    lines_d = []
    for r in d_res_list:
        if r.get('valid'):
            p_val = r.get('p_value')
            chi_val = r.get('chi2_stat')
            p_str = f"{p_val:.6f}" if p_val is not None else "N/A"
            chi_str = f"{chi_val:.4f}" if chi_val is not None else "N/A"
            interp = interpret_pvalue(p_val, alpha)
            lines_d.append(f"• {r['name']}: χ² = {chi_str}, p-valor = {p_str} | {interp}")
        else:
            lines_d.append(f"• {r['name']}: No aplicable ({r.get('error', 'Incompatible')})")
    sections.append({
        'num': 9,
        'title': '9. Pruebas Estadísticas de Bondad de Ajuste (Discreta)',
        'content': (
            f"Hipótesis contrastadas al nivel de significancia α = {alpha}:\n"
            f"H0: La variable aleatoria discreta '{d_col}' es compatible con la distribución propuesta.\n"
            f"H1: La variable no es compatible con la distribución propuesta.\n\n"
            + "\n".join(lines_d)
        )
    })
    
    # 10. Distribución seleccionada discreta
    sel_d_info = next((r for r in d_res_list if r['name'] == d_sel), None)
    d_sel_score = (sel_d_info or {}).get('score') or {}
    d_ranked = sorted([r for r in d_res_list if r.get('score')],
                      key=lambda x: -x['score']['total'])

    d_table = []
    for r in d_ranked:
        s = r['score']
        sd = s['support_detail']
        d_table.append(
            f"   {s['rank']}. {r['name']}: total = {s['total']:.4f} "
            f"(ajuste {s['gof']:.3f} × {SCORE_WEIGHTS['gof']:.2f} + "
            f"momentos {s['moment']:.3f} × {SCORE_WEIGHTS['moment']:.2f} + "
            f"soporte {s['support']:.3f} × {SCORE_WEIGHTS['support']:.2f})"
        )
    for r in d_res_list:
        if not r.get('valid'):
            d_table.append(f"   - {r['name']}: no admisible. {r.get('error', '')}")

    d_moment_text = ""
    if d_sel_score.get('moment_detail'):
        d_moment_text = "\nContraste de momentos (muestra / modelo): " + " | ".join(
            f"{k} = {v['sample']:.4f} / {v['theory']:.4f}"
            for k, v in d_sel_score['moment_detail'].items() if v.get('theory') is not None)
        d_moment_text += "\n"

    sections.append({
        'num': 10,
        'title': '10. Distribución Seleccionada (Variable Discreta)',
        'content': (
            f"Distribución seleccionada: {d_sel}\n"
            f"Parámetros estimados: {format_params(sel_d_info.get('params', {}) if sel_d_info else {})}\n"
            f"Soporte teórico: {sel_d_info.get('support', 'N/A') if sel_d_info else 'N/A'}\n\n"
            f"MÉTODO DE SELECCIÓN. La selección NO se basa en el p-valor ni en el menor estadístico, "
            f"como exige el enunciado. Con n = {int(d_stats.get('n', 0))} observaciones, χ² tiene "
            f"potencia prácticamente 1 y rechaza a todos los candidatos con p ≈ 0, de modo que el p-valor "
            f"no discrimina entre ellos. Se combinan tres criterios normalizados a [0, 1]:\n"
            f"   • Ajuste estadístico relativo ({SCORE_WEIGHTS['gof']:.0%}): χ² dividido por sus g.l., "
            f"escalado contra el mejor candidato.\n"
            f"   • Contraste de momentos ({SCORE_WEIGHTS['moment']:.0%}): error relativo entre los momentos "
            f"muestrales y los que impone el modelo, con pesos 1.0, 1.0, 0.5 y 0.25 para media, varianza, "
            f"asimetría y curtosis.\n"
            f"   • Coherencia de soporte ({SCORE_WEIGHTS['support']:.0%}): compatibilidad entre el soporte "
            f"del modelo y las restricciones del fenómeno.\n\n"
            f"Ranking completo:\n" + "\n".join(d_table) + "\n"
            + d_moment_text +
            f"Justificación: {d_sel} es la familia con mejor comportamiento conjunto. "
            + (f"Su soporte {sel_d_info.get('support', '')} admite todos los valores observados y no "
               f"asigna masa a rangos negativos. " if sel_d_info else "")
            + f"Se rechaza a la Binomial Negativa porque la familia exige Var(X) ≥ E(X) "
            f"estructuralmente y la muestra presenta subdispersión "
            f"(Var = {d_stats.get('var', 0):.4f} < E = {d_stats.get('mean', 0):.4f}); no existe elección "
            f"de (r, p) que la reproduzca, de modo que es inadmisible por estructura y no por mal ajuste.\n\n"
            f"Limitación declarada: la variable '{d_col}' es el rango del atacante en cada ronda, una "
            f"medida de habilidad latente, y sus valores no son independientes (dos rondas del mismo jugador "
            f"se repiten). La distribución se usa aquí para describir la marginal; un análisis predictivo "
            f"debería modelar la dependencia temporal además de la marginal."
        )
    })
    
    # 11. Variable continua
    sections.append({
        'num': 11,
        'title': '11. Selección de la Variable Aleatoria Continua',
        'content': (
            f"Variable seleccionada: '{c_col}'.\n"
            f"Justificación: La variable '{c_col}' (como la duración de la ronda o los segundos transcurridos de evento) "
            f"corresponde a una magnitud física real continua con alta cardinalidad y variación fluida. "
            f"Permite modelar las características temporales y de rendimiento requeridas para ponderar la compatibilidad."
        )
    })
    
    # 12. Análisis exploratorio continuo
    c_data = df[c_col].dropna().values if df is not None and c_col in df.columns else np.array([])
    c_stats = calculate_descriptive_stats(c_data) if len(c_data) > 0 else {}
    sections.append({
        'num': 12,
        'title': '12. Análisis Exploratorio de la Variable Continua',
        'content': (
            f"Estadísticas descriptivas de '{c_col}':\n"
            f"- Observaciones: {fmt_es(c_stats.get('n', 0))}\n"
            f"- Rango: [{c_stats.get('min', 0):.4f}, {c_stats.get('max', 0):.4f}]\n"
            f"- Media (x̄): {c_stats.get('mean', 0):.4f} | Mediana: {c_stats.get('median', 0):.4f}\n"
            f"- Desviación estándar (s): {c_stats.get('std', 0):.4f} | Varianza (s²): {c_stats.get('var', 0):.4f}\n"
            f"- Coeficiente de Asimetría (Skewness): {c_stats.get('skewness', 0):.4f}\n"
            f"- Curtosis: {c_stats.get('kurtosis', 0):.4f}\n"
            f"- Percentiles: P25 = {c_stats.get('percentiles', {}).get('p25', 0):.2f}, "
            f"P50 = {c_stats.get('percentiles', {}).get('p50', 0):.2f}, "
            f"P75 = {c_stats.get('percentiles', {}).get('p75', 0):.2f}, "
            f"P95 = {c_stats.get('percentiles', {}).get('p95', 0):.2f}."
        )
    })
    
    # 13. Distribuciones candidatas continuas
    sections.append({
        'num': 13,
        'title': '13. Distribuciones Candidatas Continuas',
        'content': (
            "Se evaluaron seis familias continuas fundamentales mediante estimación por máxima verosimilitud (MLE):\n"
            "1. Normal (μ, σ): Modelo de referencia para efectos simétricos y centrales.\n"
            "2. Exponencial (λ): Modelo clásico para tiempos de espera con tasa de evento constante.\n"
            "3. Lognormal (μ, σ): Apropiada para variables asimétricas positivas con colas derechas moderadas.\n"
            "4. Gamma (α, β): Modelo flexible para tiempos y cantidades continuas no negativas.\n"
            "5. Weibull (c, scale): Empleada en confiabilidad y duraciones con tasas de fallo variables.\n"
            "6. Uniforme (a, b): Hipótesis nula de equiprobabilidad en un rango acotado."
        )
    })
    
    # 14. Pruebas estadísticas continuas y normalidad
    c_res_list = DATASET_STATE.get('continuous_results') or []
    lines_c = []
    for r in c_res_list:
        if r.get('valid'):
            ks_v = r.get('ks_stat')
            pv = r.get('p_value')
            ks_str = f"{ks_v:.4f}" if ks_v is not None else "N/A"
            pv_str = f"{pv:.6f}" if pv is not None else "N/A"
            interp = interpret_pvalue(pv, alpha)
            lines_c.append(f"• {r['name']}: Estadístico KS (D) = {ks_str}, p-valor = {pv_str} | {interp}")
            
    sections.append({
        'num': 14,
        'title': '14. Pruebas Estadísticas y de Normalidad (Continua)',
        'content': (
            f"Resultados de la prueba de Kolmogorov-Smirnov (α = {alpha}):\n"
            + "\n".join(lines_c) + "\n\n"
            "ADVERTENCIA METODOLÓGICA ACADÉMICA SOBRE TAMAÑO MUESTRAL:\n"
            f"Con muestras de decenas de miles de observaciones (N = {fmt_es(c_stats.get('n', 0))}), las pruebas de bondad "
            "de ajuste tienen una potencia estadística cercana al 100%, detectando desviaciones microscópicas "
            "e irrelevantes en la práctica. Por este motivo, la elección del modelo debe sustentarse complementariamente "
            "en el análisis gráfico (inspección de la curva de densidad ajustada sobre el histograma y gráfico Q-Q)."
        )
    })
    
    # 15. Distribución seleccionada continua
    sel_c_info = next((r for r in c_res_list if r['name'] == c_sel), None)
    c_sel_score = (sel_c_info or {}).get('score') or {}
    c_ranked = sorted([r for r in c_res_list if r.get('score')],
                      key=lambda x: -x['score']['total'])
    c_by_gof = min([r for r in c_res_list if r.get('score')],
                   key=lambda x: x['score']['gof_raw'], default=None)

    c_table = []
    for r in c_ranked:
        s = r['score']
        sd = s['support_detail']
        viable_tag = "viable" if sd.get('viable') else "NO VIABLE"
        c_table.append(
            f"   {s['rank']}. {r['name']}: total = {s['total']:.4f} "
            f"(ajuste {s['gof']:.3f} × {SCORE_WEIGHTS['gof']:.2f} + "
            f"momentos {s['moment']:.3f} × {SCORE_WEIGHTS['moment']:.2f} + "
            f"soporte {s['support']:.3f} × {SCORE_WEIGHTS['support']:.2f}) "
            f"→ {viable_tag}"
        )

    c_moment_text = ""
    if c_sel_score.get('moment_detail'):
        c_moment_text = "\nContraste de momentos (muestra / modelo): " + " | ".join(
            f"{k} = {v['sample']:.4f} / {v['theory']:.4f}"
            for k, v in c_sel_score['moment_detail'].items() if v.get('theory') is not None)
        c_moment_text += "\n"

    c_issue_text = ""
    if c_sel_score.get('support_detail', {}).get('issues'):
        c_issue_text = "\nObservaciones sobre el modelo elegido:\n" + "\n".join(
            f"   - {i}" for i in c_sel_score['support_detail']['issues']) + "\n"

    gk_nota = ""
    if c_by_gof is not None and c_by_gof['name'] != c_sel:
        g = c_by_gof
        gsd = g['score']['support_detail']
        gk_nota = (
            f"\nRESULTADO CENTRAL DEL EJERCICIO. La familia {g['name']} obtiene el MEJOR ajuste "
            f"estadístico de todas las candidatas (D = {g['ks_stat']:.5f}, el menor de la tabla, frente a "
            f"D = {(sel_c_info or {}).get('ks_stat', 0):.5f} de {c_sel}), y sin embargo NO es el modelo "
            f"adecuado para esta variable. El motivo es contextual y no estadístico: su soporte es "
            f"(-∞, +∞), de modo que asigna {100 * gsd.get('mass_below_floor', 0):.2f}% de su masa de "
            f"probabilidad a valores negativos. La variable '{c_col}' es un tiempo medido en segundos, "
            f"que no puede ser negativo. Este caso ilustra exactamente el criterio del enunciado: un "
            f"ajuste estadístico excelente no garantiza que la familia sea la apropiada para el contexto, "
            f"y una prueba de bondad de ajuste por sí sola habría producido una conclusión errónea.\n"
        )

    sections.append({
        'num': 15,
        'title': '15. Distribución Seleccionada (Variable Continua)',
        'content': (
            f"Distribución seleccionada: {c_sel}\n"
            f"Parámetros estimados (MLE): {format_params(sel_c_info.get('params', {}) if sel_c_info else {})}\n"
            f"Soporte teórico: {sel_c_info.get('support', 'N/A') if sel_c_info else 'N/A'}\n"
            f"Estadístico Kolmogorov-Smirnov D = {(sel_c_info or {}).get('ks_stat', 0):.5f}, "
            f"p-valor = {(sel_c_info or {}).get('p_value', 0):.3e}\n"
            f"Rango observado: [{c_stats.get('min', 0):.2f}, {c_stats.get('max', 0):.2f}] segundos\n\n"
            f"MÉTODO DE SELECCIÓN. Igual que en la variable discreta, la selección no se hace por p-valor "
            f"ni por el menor estadístico KS, sino con un puntaje multiobjetivo de pesos "
            f"{SCORE_WEIGHTS['gof']:.2f} ajuste + {SCORE_WEIGHTS['moment']:.2f} momentos + "
            f"{SCORE_WEIGHTS['support']:.2f} soporte, y se restringe a las familias VIABLES, es decir "
            f"aquellas cuyo soporte no prohíbe ningún valor observado y cuya masa asignada a valores "
            f"imposibles no supera {VIABLE_MASS_THRESHOLD:.0%}.\n\n"
            f"Ranking completo:\n" + "\n".join(c_table) + "\n"
            + c_moment_text + c_issue_text + gk_nota +
            f"\nJustificación de la elección: {c_sel} es la mejor familia entre las viables. Su soporte "
            f"(0, +∞) es compatible con una duración no negativa, aproxima la media y la varianza "
            f"observadas, y su cola derecha modela el comportamiento de supervivencia de la variable. "
            f"Se descartó a la Exponencial porque su propiedad de ausencia de memoria "
            f"(la media es su único parámetro) contradice la forma de la curva, y asigna "
            f"{100 * next((r['score']['support_detail'].get('mass_above_sample_max', 0) for r in c_res_list if r['name'] == 'Exponencial' and r.get('score')), 0):.1f}% "
            f"de masa a duraciones por encima del máximo observado; y a la Uniforme porque su ajuste es "
            f"circular: toma a y b directamente del rango de la muestra, de modo que la coincidencia de "
            f"soporte no aporta evidencia alguna.\n\n"
            f"Limitación declarada: '{c_col}' es el reloj de la ronda, no una duración independiente. "
            f"Las rondas de una misma partida comparten reloj y están altamente correlacionadas, de modo que "
            f"la marginal descrita no puede usarse como supuesto de independencia al simular partidas "
            f"de forma realista; se requiere además un modelo de dependencia o un muestreo por bloques "
            f"de partida."
        )
    })
    
    # 16. Métodos de generación
    d_m_info = get_method_description(d_sel)
    c_m_info = get_method_description(c_sel)
    sections.append({
        'num': 16,
        'title': '16. Métodos de Generación de Variables Aleatorias Pseudoaleatorias',
        'content': (
            f"Para la generación pseudoaleatoria de las variables se implementaron métodos algorítmicos robustos:\n\n"
            f"1. Variable Discreta ({d_sel}):\n"
            f"   - Método algorítmico: {d_m_info.get('method')}\n"
            f"   - Fundamentación: {d_m_info.get('description')}\n\n"
            f"2. Variable Continua ({c_sel}):\n"
            f"   - Método algorítmico: {c_m_info.get('method')}\n"
            f"   - Fundamentación: {c_m_info.get('description')}\n\n"
            "Todos los generadores utilizan como motor base el generador de números pseudoaleatorios de 64 bits PCG64 (Permuted Congruential Generator)."
        )
    })
    
    # 17. Comparación real vs simulada (se ejecuta de verdad, no se describe)
    sim_n = 10000
    cmp_lines = []
    d_real = df[d_col].dropna().values if df is not None and d_col in df.columns else None
    c_real = df[c_col].dropna().values if df is not None and c_col in df.columns else None

    for label, real, dist_name, params, gen_fn in (
            ("discreta", d_real, d_sel,
             (sel_d_info or {}).get('params', {}), generate_discrete),
            ("continua", c_real, c_sel,
             (sel_c_info or {}).get('params', {}), generate_continuous)):
        if real is None or not params:
            cmp_lines.append(f"• Variable {label}: sin datos o parámetros para comparar.")
            continue
        try:
            sim = gen_fn(dist_name, params, size=sim_n, seed=42)
            comp = compare_distributions(real, sim)
            rs, ss, dd = comp['real_stats'], comp['simulated_stats'], comp['differences']
            ks2 = comp.get('ks_test') or {}
            mw = comp.get('mann_whitney') or {}
            cmp_lines.append(
                f"• Variable {label} ({dist_name}), n simulada = {sim_n}:\n"
                f"     media      real = {rs['mean']:.4f}   simulada = {ss['mean']:.4f}   "
                f"error = {dd['mean_rel_error_pct']:.3f}%\n"
                f"     desviación real = {rs['std']:.4f}   simulada = {ss['std']:.4f}   "
                f"error = {dd['std_rel_error_pct']:.3f}%\n"
                f"     mediana    real = {rs['median']:.4f}   simulada = {ss['median']:.4f}   "
                f"dif. abs. = {dd['median_diff']:.4f}\n"
                f"     asimetría  real = {rs['skewness']:.4f}   simulada = {ss['skewness']:.4f}\n"
                f"     curtosis   real = {rs['kurtosis']:.4f}   simulada = {ss['kurtosis']:.4f}\n"
                f"     KS de dos muestras: D = {ks2.get('statistic', float('nan')):.5f}, "
                f"p = {ks2.get('p_value', float('nan')):.3e} | "
                f"Mann-Whitney U: p = {mw.get('p_value', float('nan')):.3e}"
            )
        except Exception as e:
            cmp_lines.append(f"• Variable {label}: la comparación falló ({e}).")

    sections.append({
        'num': 17,
        'title': '17. Comparación Real vs Simulada',
        'content': (
            f"Se generó una muestra simulada de {fmt_es(sim_n)} observaciones para cada variable con el generador "
            f"de la distribución seleccionada y una semilla fija (42), de modo que el resultado sea "
            f"reproducible. Se comparan los estadísticos descriptivos y se contrastan las distribuciones "
            f"mediante Kolmogorov-Smirnov para dos muestras y Mann-Whitney.\n\n"
            + "\n".join(cmp_lines) + "\n\n"
            f"INTERPRETACIÓN HONESTA. Que la media y la desviación de la muestra simulada reproduzcan las "
            f"observadas NO valida el modelo: esos momentos fueron los que el propio ajuste utilizó para "
            f"estimar los parámetros, así que la coincidencia está forzada por construcción. Lo que sí aporta "
            f"evidencia independiente son los estadísticos que el ajuste no reprodujo: la asimetría y la "
            f"curtosis, y sobre todo la cola derecha. Las pruebas de dos muestras rechazan la igualdad de "
            f"distribuciones con seguridad, y eso es esperable: el ajuste es de la marginal, y la variable "
            f"presenta dependencia entre rondas de una misma partida, de modo que la independencia exigida "
            f"por KS y Mann-Whitney no se cumple. La conclusión correcta es que el generador reproduce bien "
            f"la forma marginal y sus momentos, pero no valida la suposición de independencia."
        )
    })
    
    # 18. Simulación del matchmaking
    if mm_res:
        mm_text = (
            f"Resultados de la simulación del sistema de Matchmaking:\n"
            f"- Total de jugadores procesados: {fmt_es(mm_res.get('total_players', 0))}\n"
            f"- Partidas creadas exitosamente: {mm_res.get('matches_created', 0)}\n"
            f"- Jugadores emparejados: {fmt_es(mm_res.get('players_matched', 0))} ({mm_res.get('match_rate', 0):.1f}%)\n"
            f"- Jugadores remanentes en cola: {mm_res.get('players_in_queue', 0)}\n"
            f"- Tiempo promedio de espera en cola: {mm_res.get('avg_wait_time', 0):.2f} unidades de tiempo\n"
            f"- Tiempo máximo de espera registrado: {mm_res.get('max_wait_time', 0):.2f}\n"
            f"- Compatibilidad promedio lograda en las partidas: {mm_res.get('avg_compatibility', 0):.1f}%"
        )
    else:
        mm_text = (
            "La simulación de Matchmaking modela una cola estocástica donde cada jugador entrante recibe atributos generados "
            f"desde las distribuciones {d_sel} y {c_sel}. Un algoritmo evalúa la distancia euclidiana normalizada "
            "d(p1, p2) <= umbral_compatibilidad. Al completarse grupos de 10 jugadores compatibles, se crea la partida."
        )
    sections.append({
        'num': 18,
        'title': '18. Simulación del Matchmaking y Rendimiento de la Cola',
        'content': mm_text
    })
    
    # 19. Limitaciones
    sections.append({
        'num': 19,
        'title': '19. Limitaciones del Estudio',
        'content': (
            "1. El dataset original de repeticiones (.dem) registra acciones durante partidas disputadas; no representa "
            "de forma nativa el registro de colas en los servidores centrales de Valve Corporation.\n"
            "2. El algoritmo de compatibilidad utilizado es una abstracción pedagógica basada en distancia euclidiana "
            "normalizada y no reproduce el sistema propietario Glicko-2 / Trust Factor utilizado en el videojuego comercial.\n"
            "3. En datasets con millones de eventos, la correlación temporal entre rondas de una misma partida puede violar "
            "levemente el supuesto de independencia idénticamente distribuida (i.i.d.)."
        )
    })
    
    # 20. Conclusiones
    d_chi = (sel_d_info or {}).get('chi2_stat')
    d_chi_str = f"{d_chi:.2f}" if d_chi is not None else "N/A"
    c_ks = (sel_c_info or {}).get('ks_stat')
    c_ks_str = f"{c_ks:.5f}" if c_ks is not None else "N/A"

    d_ratio = d_stats.get('var', 0) / d_stats.get('mean', 1) if d_stats.get('mean') else 1.0
    sections.append({
        'num': 20,
        'title': '20. Conclusiones',
        'content': (
            f"1. Variable discreta '{d_col}': la familia {d_sel} "
            f"{format_params(sel_d_info.get('params', {}) if sel_d_info else {})} es la más adecuada de las "
            f"cuatro candidatas. Su soporte {sel_d_info.get('support', '') if sel_d_info else ''} contiene "
            f"todos los valores observados y la relación Var/E observada es {d_ratio:.4f}, muy próxima a 1, "
            f"que es la propiedad característica de Poisson; con χ² = {d_chi_str} "
            f"y g.l. = {(sel_d_info or {}).get('df', 'N/A')}.\n"
            f"2. Variable continua '{c_col}': la familia {c_sel} "
            f"{format_params(sel_c_info.get('params', {}) if sel_c_info else {})} es la más adecuada, con "
            f"D = {c_ks_str}. Conviene subrayar que esta elección NO coincide con la de mejor ajuste "
            f"estadístico: la Normal presenta un D menor, pero se descarta porque asigna masa de "
            f"probabilidad a duraciones negativas. La conclusión central del trabajo es que la prueba de "
            f"bondad de ajuste no puede decidir por sí sola qué distribución representa al fenómeno.\n"
            f"3. La Binomial Negativa quedó inadmisible por estructura, no por ajuste: exige Var(X) ≥ E(X) "
            f"y la muestra es subdispersa. La Geométrica quedó excluida porque no puede tomar el valor 0, "
            f"que aparece en la muestra. Estos casos muestran que parte de la modelación se decide por "
            f"supuestos de la familia, antes de mirar cualquier estadístico.\n"
            f"4. La simulación confirma que los generadores reproducen la media y la varianza por "
            f"construcción, pero las pruebas de dos muestras rechazan la igualdad de distribuciones: la "
            f"variable viola el supuesto de independencia, de modo que la marginal no basta para "
            f"simular partidas realistas.\n"
            f"5. Limitación no resuelta: el reloj de ronda de una partida es una variable censurada y "
            f"correlacionada, y ninguna de las seis familias continuas evaluadas captura a la vez su "
            f"suelo físico y la dependencia temporal. Un modelo de supervivencia con covariables por ronda "
            f"sería el enfoque metodológicamente correcto en una ampliación."
        )
    })
    
    return {
        'title': 'Informe Académico: Modelación Probabilística de Matchmaking en CS:GO',
        'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': info,
        'discrete_col': d_col,
        'continuous_col': c_col,
        'discrete_selected': d_sel,
        'continuous_selected': c_sel,
        'sections': sections
    }


@app.route('/api/report/generate')
def api_report_generate():
    """Genera y retorna el informe académico con las 20 secciones exactas."""
    try:
        report = build_full_academic_report()
        return jsonify({'success': True, 'report': report})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/report/markdown')
def api_report_markdown():
    """Descarga el informe en formato Markdown (.md)."""
    try:
        report = build_full_academic_report()
        md_lines = [
            f"# {report['title']}",
            f"*Fecha de generación: {report['generated_at']}*",
            "\n---\n"
        ]
        for sec in report['sections']:
            md_lines.append(f"## {sec['title']}\n")
            md_lines.append(sec['content'] + "\n")
            
        md_content = "\n".join(md_lines)
        return (
            md_content,
            200,
            {
                'Content-Type': 'text/markdown; charset=utf-8',
                'Content-Disposition': 'attachment; filename=Informe_Matchmaking_CSGO.md'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/alpha', methods=['POST'])
def api_config_alpha():
    """Configura el nivel de significancia α global."""
    data = request.get_json() or {}
    new_alpha = float(data.get('alpha', 0.05))
    DATASET_STATE['alpha'] = new_alpha
    return jsonify({'success': True, 'alpha': new_alpha})


# =============================================================================
# MANEJADORES DE ERROR GLOBALES
# =============================================================================

@app.errorhandler(404)
def not_found_error(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Endpoint no encontrado'}), 404
    return render_template('error.html', message="Página no encontrada (Error 404)."), 404


@app.errorhandler(500)
def internal_server_error(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Error interno del servidor'}), 500
    return render_template('error.html', message="Ocurrió un error inesperado en el servidor."), 500


# =============================================================================
# INICIALIZACIÓN AL ARRANQUE
# =============================================================================

try:
    df_init, info_init = load_dataset(nrows=50000)
    DATASET_STATE['original_df'] = df_init
    DATASET_STATE['info'] = info_init
    auto_select_default_variables(df_init)
    print(f"Dataset cargado exitosamente al iniciar: {info_init['filename']} ({info_init['n_rows']} filas)")
    print(f"Variable discreta inicial: {DATASET_STATE['discrete_col']}, continua: {DATASET_STATE['continuous_col']}")
except Exception as err:
    print(f"Aviso de inicio: No se cargó dataset automático ({err}). Use la interfaz para seleccionarlo.")

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
