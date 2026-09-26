"""
Módulo de pruebas estadísticas y cálculos para el Simulador de Matchmaking CS:GO.
Incluye pruebas de bondad de ajuste, pruebas de normalidad, pruebas de dos muestras
y utilidades para gráficos estadísticos (Q-Q plot, ECDF).
"""

import numpy as np
import pandas as pd
from scipy import stats


def chi_square_test(observed_freq, expected_freq, ddof=0):
    """
    Prueba de Chi-cuadrado para bondad de ajuste en variables discretas.
    Agrupa automáticamente colas con frecuencias esperadas < 5 para garantizar
    la validez matemática de la aproximación asintótica.
    
    Args:
        observed_freq: Lista o array de frecuencias observadas (Oi).
        expected_freq: Lista o array de frecuencias esperadas (Ei).
        ddof: Grados de libertad restados por parámetros estimados (p. ej., 1 para Poisson).
        
    Returns:
        Diccionario con estadístico chi2, p-valor, grados de libertad y validez.
    """
    obs = np.array(observed_freq, dtype=float)
    exp = np.array(expected_freq, dtype=float)
    
    # Filtrar valores no positivos o nulos
    valid_mask = (exp > 0) & np.isfinite(exp) & np.isfinite(obs)
    obs = obs[valid_mask]
    exp = exp[valid_mask]
    
    if len(obs) < 2:
        return {
            'statistic': float(np.nan),
            'p_value': float(np.nan),
            'df': 0,
            'valid': False,
            'pooled_bins': 0,
            'message': 'Insuficientes categorías para la prueba Chi-cuadrado'
        }
    
    # Agrupamiento de clases (bin pooling) para Ei >= 5 cuando sea posible
    pooled_obs = []
    pooled_exp = []
    
    cur_obs = 0.0
    cur_exp = 0.0
    
    for o, e in zip(obs, exp):
        cur_obs += o
        cur_exp += e
        if cur_exp >= 5.0:
            pooled_obs.append(cur_obs)
            pooled_exp.append(cur_exp)
            cur_obs = 0.0
            cur_exp = 0.0
            
    # Añadir remanente al último bin si quedó algo
    if cur_exp > 0:
        if len(pooled_obs) > 0:
            pooled_obs[-1] += cur_obs
            pooled_exp[-1] += cur_exp
        else:
            pooled_obs.append(cur_obs)
            pooled_exp.append(cur_exp)
            
    # Si tras agrupar quedan menos de 2 bins, usar los originales para reportar el estadístico
    if len(pooled_obs) >= 2:
        final_obs = np.array(pooled_obs)
        final_exp = np.array(pooled_exp)
    else:
        final_obs = obs
        final_exp = exp
        
    # Normalizar frecuencias esperadas para que sumen exactamente la suma de observadas
    sum_obs = np.sum(final_obs)
    sum_exp = np.sum(final_exp)
    if sum_exp > 0:
        final_exp = final_exp * (sum_obs / sum_exp)
        
    chi2_stat = np.sum((final_obs - final_exp) ** 2 / final_exp)
    k = len(final_obs)
    degrees_of_freedom = max(1, k - 1 - ddof)
    
    p_value = 1.0 - stats.chi2.cdf(chi2_stat, degrees_of_freedom)
    
    return {
        'statistic': float(chi2_stat),
        'p_value': float(p_value),
        'df': int(degrees_of_freedom),
        'k_bins': int(k),
        'valid': True,
        'message': f'Prueba Chi-cuadrado calculada con {k} categorías ({degrees_of_freedom} g.l.)'
    }


def ks_test(data, dist_name, params):
    """
    Prueba de Kolmogorov-Smirnov para distribuciones continuas.
    Compara la función de distribución acumulada empírica con la teórica ajustada.
    
    Args:
        data: Array de datos continuos observados.
        dist_name: Nombre de la distribución scipy (p. ej. 'norm', 'expon', 'lognorm').
        params: Parámetros estimados de la distribución.
        
    Returns:
        Diccionario con estadístico D y p-valor.
    """
    try:
        clean_data = np.array(data, dtype=float)
        clean_data = clean_data[np.isfinite(clean_data)]
        
        if len(clean_data) < 5:
            return {'statistic': float(np.nan), 'p_value': float(np.nan), 'valid': False, 'error': 'Muestra insuficiente'}
            
        dist = getattr(stats, dist_name)
        
        # Ajustar parámetros con scipy si no están directamente en la firma requerida
        if dist_name == 'norm':
            res = stats.kstest(clean_data, 'norm', args=(params.get('mu', np.mean(clean_data)), params.get('sigma', np.std(clean_data))))
        elif dist_name == 'expon':
            res = stats.kstest(clean_data, 'expon', args=(0, params.get('scale', np.mean(clean_data))))
        elif dist_name == 'lognorm':
            res = stats.kstest(clean_data, 'lognorm', args=(params.get('sigma', 1.0), 0, np.exp(params.get('mu', 0.0))))
        elif dist_name == 'gamma':
            res = stats.kstest(clean_data, 'gamma', args=(params.get('shape', 1.0), 0, params.get('scale', 1.0)))
        elif dist_name == 'weibull_min':
            res = stats.kstest(clean_data, 'weibull_min', args=(params.get('shape', 1.0), 0, params.get('scale', 1.0)))
        elif dist_name == 'uniform':
            a = params.get('a', np.min(clean_data))
            b = params.get('b', np.max(clean_data))
            res = stats.kstest(clean_data, 'uniform', args=(a, b - a))
        else:
            fitted_params = dist.fit(clean_data)
            res = stats.kstest(clean_data, lambda x: dist.cdf(x, *fitted_params))
            
        return {
            'statistic': float(res.statistic),
            'p_value': float(res.pvalue),
            'valid': True
        }
    except Exception as e:
        return {'statistic': float(np.nan), 'p_value': float(np.nan), 'valid': False, 'error': str(e)}


def shapiro_test(data, max_samples=5000):
    """
    Prueba de normalidad de Shapiro-Wilk.
    Apropiada para muestras pequeñas o moderadas. Si n > max_samples, toma una
    submuestra aleatoria reproducible y documenta la metodología.
    
    Args:
        data: Datos numéricos.
        max_samples: Tamaño máximo de muestra permitido por scipy.stats.shapiro.
        
    Returns:
        Diccionario con estadístico W, p-valor y nota metodológica.
    """
    try:
        clean_data = np.array(data, dtype=float)
        clean_data = clean_data[np.isfinite(clean_data)]
        n = len(clean_data)
        
        if n < 3:
            return {'statistic': float(np.nan), 'p_value': float(np.nan), 'valid': False, 'error': 'Muestra insuficiente (n < 3)'}
            
        note = None
        if n > max_samples:
            rng = np.random.default_rng(42)
            sample = rng.choice(clean_data, max_samples, replace=False)
            note = f"Se evaluó una submuestra aleatoria de {max_samples:,} observaciones (límite del test de Shapiro-Wilk)."
        else:
            sample = clean_data
            
        stat, p_val = stats.shapiro(sample)
        return {
            'statistic': float(stat),
            'p_value': float(p_val),
            'valid': True,
            'n_tested': len(sample),
            'note': note
        }
    except Exception as e:
        return {'statistic': float(np.nan), 'p_value': float(np.nan), 'valid': False, 'error': str(e)}


def anderson_test(data):
    """
    Prueba de normalidad de Anderson-Darling.
    Da mayor peso a las colas de la distribución que la prueba KS.
    
    Args:
        data: Datos numéricos.
        
    Returns:
        Diccionario con estadístico A2, valores críticos y niveles de significancia.
    """
    try:
        clean_data = np.array(data, dtype=float)
        clean_data = clean_data[np.isfinite(clean_data)]
        
        if len(clean_data) < 5:
            return {'statistic': float(np.nan), 'valid': False, 'error': 'Muestra insuficiente'}
            
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            res = stats.anderson(clean_data, dist='norm')
        
        # Determinar si se rechaza a alpha=0.05
        # res.significance_level típicamente [15.0, 10.0, 5.0, 2.5, 1.0]
        # el índice correspondiente a 5% es 2
        crit_5 = None
        reject_5 = False
        if 5.0 in res.significance_level:
            idx = list(res.significance_level).index(5.0)
            crit_5 = float(res.critical_values[idx])
            reject_5 = bool(res.statistic > crit_5)
            
        return {
            'statistic': float(res.statistic),
            'critical_values': [float(x) for x in res.critical_values],
            'significance_levels': [float(x) for x in res.significance_level],
            'crit_val_5pct': crit_5,
            'reject_at_5pct': reject_5,
            'valid': True
        }
    except Exception as e:
        return {'statistic': float(np.nan), 'valid': False, 'error': str(e)}


def dagostino_test(data):
    """
    Prueba de normalidad de D'Agostino-Pearson (K^2).
    Combina pruebas de asimetría (skewness) y curtosis (kurtosis).
    
    Args:
        data: Datos numéricos.
        
    Returns:
        Diccionario con estadístico K2 y p-valor.
    """
    try:
        clean_data = np.array(data, dtype=float)
        clean_data = clean_data[np.isfinite(clean_data)]
        
        if len(clean_data) < 20:
            return {'statistic': float(np.nan), 'p_value': float(np.nan), 'valid': False, 'error': 'Requiere al menos 20 observaciones'}
            
        stat, p_val = stats.normaltest(clean_data)
        return {
            'statistic': float(stat),
            'p_value': float(p_val),
            'valid': True
        }
    except Exception as e:
        return {'statistic': float(np.nan), 'p_value': float(np.nan), 'valid': False, 'error': str(e)}


def ks_2samp_test(data1, data2):
    """
    Prueba de Kolmogorov-Smirnov para dos muestras independientes.
    Contrasta si los datos reales y los datos simulados provienen de la misma distribución continua.
    
    H0: Ambas muestras provienen de la misma distribución continua.
    H1: Las muestras provienen de distribuciones distintas.
    
    Args:
        data1: Primera muestra (p. ej. datos reales).
        data2: Segunda muestra (p. ej. datos simulados).
        
    Returns:
        Diccionario con estadístico D y p-valor.
    """
    try:
        d1 = np.array(data1, dtype=float)
        d2 = np.array(data2, dtype=float)
        d1 = d1[np.isfinite(d1)]
        d2 = d2[np.isfinite(d2)]
        
        if len(d1) == 0 or len(d2) == 0:
            return {'statistic': float(np.nan), 'p_value': float(np.nan), 'valid': False}
            
        res = stats.ks_2samp(d1, d2)
        return {
            'statistic': float(res.statistic),
            'p_value': float(res.pvalue),
            'valid': True
        }
    except Exception as e:
        return {'statistic': float(np.nan), 'p_value': float(np.nan), 'valid': False, 'error': str(e)}


def mann_whitney_test(data1, data2):
    """
    Prueba no paramétrica de Mann-Whitney U para comparar dos muestras independientes.
    Contrasta si una muestra tiende a tomar valores mayores que la otra.
    
    Args:
        data1: Muestra real.
        data2: Muestra simulada.
        
    Returns:
        Diccionario con estadístico U y p-valor.
    """
    try:
        d1 = np.array(data1, dtype=float)
        d2 = np.array(data2, dtype=float)
        d1 = d1[np.isfinite(d1)]
        d2 = d2[np.isfinite(d2)]
        
        if len(d1) == 0 or len(d2) == 0:
            return {'statistic': float(np.nan), 'p_value': float(np.nan), 'valid': False}
            
        # Si las muestras son muy grandes, submuestrear a 10,000 para rapidez computacional
        if len(d1) > 10000:
            rng = np.random.default_rng(42)
            d1 = rng.choice(d1, 10000, replace=False)
        if len(d2) > 10000:
            rng = np.random.default_rng(42)
            d2 = rng.choice(d2, 10000, replace=False)
            
        res = stats.mannwhitneyu(d1, d2, alternative='two-sided')
        return {
            'statistic': float(res.statistic),
            'p_value': float(res.pvalue),
            'valid': True
        }
    except Exception as e:
        return {'statistic': float(np.nan), 'p_value': float(np.nan), 'valid': False, 'error': str(e)}


def qq_plot_data(data, dist_name='norm', n_points=100):
    """
    Calcula los cuantiles teóricos vs empíricos para generar un Q-Q plot con Plotly.js.
    
    Args:
        data: Datos empíricos.
        dist_name: Distribución de referencia ('norm').
        n_points: Número de cuantiles a interpolar.
        
    Returns:
        Diccionario con theoretical_quantiles, sample_quantiles, line_x, line_y.
    """
    clean_data = np.array(data, dtype=float)
    clean_data = clean_data[np.isfinite(clean_data)]
    
    if len(clean_data) < 5:
        return {'theoretical': [], 'sample': [], 'line_x': [], 'line_y': []}
        
    # Estandarizar datos para comparar con N(0, 1)
    mean = np.mean(clean_data)
    std = np.std(clean_data)
    std_data = (clean_data - mean) / std if std > 0 else clean_data - mean
    
    # Cuantiles uniformemente espaciados
    probs = np.linspace(0.01, 0.99, n_points)
    theoretical_quantiles = stats.norm.ppf(probs)
    sample_quantiles = np.quantile(std_data, probs)
    
    min_val = min(theoretical_quantiles[0], sample_quantiles[0])
    max_val = max(theoretical_quantiles[-1], sample_quantiles[-1])
    
    return {
        'theoretical': [float(x) for x in theoretical_quantiles],
        'sample': [float(x) for x in sample_quantiles],
        'line_x': [float(min_val), float(max_val)],
        'line_y': [float(min_val), float(max_val)]
    }


def interpret_pvalue(p_val, alpha=0.05, test_name="Prueba de hipótesis"):
    """
    Proporciona una interpretación formal y académica rigurosa del p-valor.
    Nunca afirma que p > alpha 'demuestra' la veracidad de la hipótesis nula.
    
    Args:
        p_val: p-valor obtenido en la prueba.
        alpha: Nivel de significancia establecido (por defecto 0.05).
        test_name: Nombre o tipo de prueba.
        
    Returns:
        Cadena con la interpretación académica.
    """
    if p_val is None or np.isnan(p_val):
        return "No fue posible calcular el p-valor debido a restricciones en los parámetros o tamaño muestral."
        
    if p_val >= alpha:
        return (
            f"Como p ({p_val:.4f}) ≥ α ({alpha}), no se rechaza la hipótesis nula H0. "
            f"No existe evidencia estadística suficiente para descartar que los datos son compatibles con la distribución propuesta."
        )
    else:
        return (
            f"Como p ({p_val:.4f}) < α ({alpha}), se rechaza la hipótesis nula H0 al nivel de significancia α={alpha}. "
            f"Existe evidencia estadística de discrepancia entre los datos observados y el modelo teórico. "
            f"(Nota académica: en muestras de gran tamaño, desviaciones prácticas mínimas pueden resultar estadísticamente significativas)."
        )
