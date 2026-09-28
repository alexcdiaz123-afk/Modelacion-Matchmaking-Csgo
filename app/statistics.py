"""
Módulo de pruebas estadísticas y cálculos para el Simulador de Matchmaking CS:GO.
Incluye pruebas de bondad de ajuste, pruebas de normalidad, pruebas de dos muestras
y utilidades para gráficos estadísticos (Q-Q plot, ECDF).
"""

import math
from functools import partial

import numpy as np
import pandas as pd
from scipy import stats


def chi_square_test(observed_freq, expected_freq, ddof=0):
    """
    Prueba de Chi-cuadrado para bondad de ajuste en variables discretas.

    Aplica dos correcciones metodológicas necesarias para que el estadístico sea
    interpretable:

    1. COLA NO OBSERVADA: si la suma de las frecuencias esperadas sobre el soporte
       observado es menor que N, la masa que corresponde a valores no presentes en
       la muestra (por ejemplo las rondas 19..30 en un rango de habilidad acotado
       a 18) se agrega como una categoría adicional con O = 0 y E = déficit. Esta es
       la masa de probabilidad que el modelo asigna al soporte y que la muestra no
       respalda; omitirla (o redistribuirla) sesgaría el estadístico.

    2. AGRUPAMIENTO DE COLAS: se acumulan categorías adyacentes hasta alcanzar
       E >= 5, condición de validez de la aproximación asintótica de Pearson.

    No se renormaliza el vector esperado: hacerlo (escalar E para que sume N)
    reduce artificialmente el estadístico y sesga la decisión.

    Args:
        observed_freq: Lista o array de frecuencias observadas (Oi).
        expected_freq: Lista o array de frecuencias esperadas (Ei).
        ddof: Grados de libertad restados por parámetros estimados (p. ej., 1 para Poisson).

    Returns:
        Diccionario con estadístico chi2, p-valor, grados de libertad y validez.
    """
    obs = np.array(observed_freq, dtype=float)
    exp = np.array(expected_freq, dtype=float)

    # Descartar categorías con esperado no positivo o no finito
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

    total_obs = float(np.sum(obs))

    # 1. Categoría explícita para la cola no observada (soporte del modelo > soporte muestral)
    tail_mass = total_obs - float(np.sum(exp))
    if tail_mass > 0.5:
        obs = np.append(obs, 0.0)
        exp = np.append(exp, tail_mass)

    # 2. Agrupamiento de clases adyacentes hasta alcanzar E >= 5
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

    # El remanente se une al último bin para no perder masa de probabilidad
    if cur_exp > 0.5:
        if pooled_obs:
            pooled_obs[-1] += cur_obs
            pooled_exp[-1] += cur_exp
        else:
            pooled_obs.append(cur_obs)
            pooled_exp.append(cur_exp)

    if len(pooled_obs) >= 2:
        final_obs = np.array(pooled_obs)
        final_exp = np.array(pooled_exp)
    else:
        final_obs = obs
        final_exp = exp

    # Sin renormalización: el estadístico usa las frecuencias tal como quedaron
    with np.errstate(divide='ignore', invalid='ignore'):
        contributions = np.where(final_exp > 0, (final_obs - final_exp) ** 2 / final_exp, 0.0)
    chi2_stat = float(np.sum(contributions))

    k = len(final_obs)
    degrees_of_freedom = max(1, k - 1 - ddof)
    p_value = float(1.0 - stats.chi2.cdf(chi2_stat, degrees_of_freedom))

    return {
        'statistic': chi2_stat,
        'p_value': p_value,
        'df': int(degrees_of_freedom),
        'k_bins': int(k),
        'n_categories_raw': int(len(obs)),
        'tail_mass': float(max(0.0, tail_mass)),
        'valid': True,
        'message': f'Chi-cuadrado con {k} clases agrupadas ({degrees_of_freedom} g.l.)'
    }


def build_cdf(dist_name, params):
    """
    Construye la función de distribución acumulada teórica F(x) a partir del nombre
    de la distribución de SciPy y de los parámetros estimados.

    Se devuelve siempre un *callable* y no el par (nombre, args) porque el despacho
    por nombre de `scipy.stats.kstest` invoca internamente `special.ndtr(x, loc, scale)`
    para 'norm', lo que produce el error
    `ndtr() takes from 1 to 2 positional arguments but 3 were given`
    en SciPy >= 1.18 y devuelve un NaN silencioso. El camino con callable es
    equivalente (verificado contra el despacho por nombre en las demás familias)
    y es inmune a ese fallo.

    Args:
        dist_name: Nombre de la distribución en scipy.stats ('norm', 'expon', ...).
        params: Diccionario con los parámetros estimados.

    Returns:
        Callable F(x) -> P(X <= x), o None si la combinación no es válida.
    """
    if dist_name == 'norm':
        return partial(stats.norm.cdf,
                       loc=params.get('mu', 0.0),
                       scale=params.get('sigma', 1.0))
    if dist_name == 'expon':
        return partial(stats.expon.cdf,
                       loc=0.0,
                       scale=params.get('scale', params.get('lambda', 1.0) and 1.0 / params['lambda']))
    if dist_name == 'lognorm':
        return partial(stats.lognorm.cdf,
                       s=params.get('sigma', 1.0),
                       loc=0.0,
                       scale=params.get('scale', math.exp(params.get('mu', 0.0))))
    if dist_name == 'gamma':
        return partial(stats.gamma.cdf,
                       a=params.get('shape', 1.0),
                       loc=0.0,
                       scale=params.get('scale', 1.0))
    if dist_name == 'weibull_min':
        return partial(stats.weibull_min.cdf,
                       c=params.get('shape', 1.0),
                       loc=0.0,
                       scale=params.get('scale', 1.0))
    if dist_name == 'uniform':
        a = params.get('a', 0.0)
        b = params.get('b', 1.0)
        if not b > a:
            return None
        return partial(stats.uniform.cdf, loc=a, scale=b - a)
    if dist_name == 'poisson':
        return lambda x: np.asarray([stats.poisson.cdf(int(v), params.get('lambda', 1.0)) for v in np.atleast_1d(x)])
    if dist_name == 'nbinom':
        return lambda x: np.asarray([stats.nbinom.cdf(int(v), params.get('r', 1.0), params.get('p', 0.5)) for v in np.atleast_1d(x)])
    if dist_name == 'binom':
        return lambda x: np.asarray([stats.binom.cdf(int(v), int(params.get('n', 10)), params.get('p', 0.5)) for v in np.atleast_1d(x)])
    if dist_name == 'geom':
        return lambda x: np.asarray([stats.geom.cdf(int(v), params.get('p', 0.5)) for v in np.atleast_1d(x)])
    return None


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

        cdf = build_cdf(dist_name, params)
        if cdf is None:
            return {'statistic': float(np.nan), 'p_value': float(np.nan), 'valid': False,
                    'error': f"Parámetros inválidos para '{dist_name}'"}

        res = stats.kstest(clean_data, cdf)

        if not np.isfinite(res.statistic):
            return {'statistic': float(np.nan), 'p_value': float(np.nan), 'valid': False,
                    'error': 'El estadístico KS no pudo calcularse'}

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
