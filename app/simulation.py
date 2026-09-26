"""
Módulo de generación de variables aleatorias simuladas y comparación estadística.
Documenta los métodos matemáticos y algorítmicos utilizados para la simulación
(transformada inversa, Box-Muller, Marsaglia-Tsang, BTPE, etc.).
"""

import numpy as np
import pandas as pd
from scipy import stats
from statistics import ks_2samp_test, mann_whitney_test


# =============================================================================
# DOCUMENTACIÓN DE MÉTODOS DE GENERACIÓN
# =============================================================================

DISCRETE_METHODS = {
    'Poisson': {
        'method': 'Algoritmo de Knuth y Transformación por Rechazo',
        'academic_explanation': (
            "Para tasas pequeñas (λ < 30), NumPy implementa el algoritmo de Knuth: "
            "se generan variables uniformes independientes U_1, U_2, ... hasta que "
            "el producto ∏ U_i < e^(-λ), donde el valor retornado es el índice k-1. "
            "Para λ >= 30, se emplea el método de transformación por rechazo gaussiano/lorentziano "
            "para garantizar eficiencia temporal O(1)."
        ),
        'numpy_func': lambda lam, size, rng: rng.poisson(lam, size)
    },
    'Binomial': {
        'method': 'Algoritmo BTPE (Binomial, Triangle, Parallelogram, Exponential)',
        'academic_explanation': (
            "NumPy emplea el algoritmo BTPE (Kachitvichyanukul & Schmeiser), un método de aceptación-rechazo "
            "que divide el histograma de la distribución binomial en cuatro regiones (triángulo central, "
            "paralelogramo y dos colas exponenciales). Para tamaños de muestra pequeños (n*p < 30), "
            "se suma directamente una secuencia de n variables de Bernoulli independientes."
        ),
        'numpy_func': lambda n, p, size, rng: rng.binomial(n, p, size)
    },
    'Binomial Negativa': {
        'method': 'Mezcla Poisson-Gamma / Transformación Inversa',
        'academic_explanation': (
            "Se fundamenta en la representación de la Binomial Negativa como una mezcla de Poisson con tasa λ "
            "distribuida según una Gamma(r, (1-p)/p). Primero se simula Y ~ Gamma(r, scale=(1-p)/p) y "
            "luego X ~ Poisson(Y), lo que genera la sobredispersión característica de forma exacta."
        ),
        'numpy_func': lambda r, p, size, rng: rng.negative_binomial(r, p, size)
    },
    'Geométrica': {
        'method': 'Método de la Transformada Inversa',
        'academic_explanation': (
            "Se aplica la función inversa de la distribución acumulada F(x) = 1 - (1-p)^x. "
            "Generando una variable pseudoaleatoria continua U ~ Uniforme(0, 1), el valor simulado "
            "se obtiene mediante X = ceil(ln(1 - U) / ln(1 - p))."
        ),
        'numpy_func': lambda p, size, rng: rng.geometric(p, size)
    },
}

CONTINUOUS_METHODS = {
    'Normal': {
        'method': 'Transformada de Box-Muller / Algoritmo Ziggurat',
        'academic_explanation': (
            "NumPy implementa de forma optimizada el algoritmo Ziggurat (Marsaglia & Tsang), el cual "
            "cubre la campana de Gauss mediante rectángulos horizontales apilados (el zigurat) "
            "con aceptación directa en más del 98% de los casos. Teóricamente equivale a la transformada "
            "de Box-Muller: Z_0 = sqrt(-2 ln U_1) * cos(2π U_2) con escalamiento X = μ + σ Z_0."
        ),
        'numpy_func': lambda mu, sigma, size, rng: rng.normal(mu, sigma, size)
    },
    'Exponencial': {
        'method': 'Método de la Transformada Inversa',
        'academic_explanation': (
            "Dado que la función de distribución acumulada es F(x) = 1 - e^(-x/β), su inversa analítica "
            "es F^(-1)(u) = -β * ln(1 - u). Siendo U ~ Uniforme(0, 1), como 1-U también es Uniforme(0, 1), "
            "se evalúa X = -β * ln(U)."
        ),
        'numpy_func': lambda scale, size, rng: rng.exponential(scale, size)
    },
    'Lognormal': {
        'method': 'Transformación Exponencial de Variable Normal',
        'academic_explanation': (
            "Por definición, si Y ~ Normal(μ, σ^2), entonces X = exp(Y) sigue una distribución Lognormal. "
            "El generador simula primero la componente normal mediante el algoritmo Ziggurat y luego "
            "aplica la función exponencial componente a componente."
        ),
        'numpy_func': lambda mu, sigma, size, rng: rng.lognormal(mu, sigma, size)
    },
    'Gamma': {
        'method': 'Algoritmo de Marsaglia y Tsang (Aceptación-Rechazo)',
        'academic_explanation': (
            "Para parámetro de forma shape >= 1, se emplea el método de Marsaglia-Tsang, que transforma "
            "una variable normal en una variable gamma cúbica con ajuste por rechazo. "
            "Para shape < 1, se aplica la propiedad de descomposición: si X ~ Gamma(shape + 1), "
            "entonces X * U^(1/shape) ~ Gamma(shape)."
        ),
        'numpy_func': lambda shape, scale, size, rng: rng.gamma(shape, scale, size)
    },
    'Weibull': {
        'method': 'Método de la Transformada Inversa',
        'academic_explanation': (
            "La función de distribución acumulada es F(x) = 1 - exp(-(x / scale)^shape). "
            "Igualando a U ~ Uniforme(0, 1) e invirtiendo la función analíticamente, "
            "se calcula directamente X = scale * (-ln(U))^(1 / shape)."
        ),
        'numpy_func': lambda shape, scale, size, rng: scale * rng.weibull(shape, size)
    },
    'Uniforme': {
        'method': 'Transformación Lineal del Generador Congruencial / PCG64',
        'academic_explanation': (
            "El generador pseudoaleatorio base de NumPy (PCG64) produce enteros uniformes de 64 bits "
            "divididos por 2^64 para mapear al intervalo continuo [0, 1). Posteriormente se aplica "
            "el escalamiento lineal afín: X = a + (b - a) * U."
        ),
        'numpy_func': lambda a, b, size, rng: rng.uniform(a, b, size)
    },
}


def generate_discrete(dist_name, params, size=10000, seed=42):
    """
    Genera observaciones pseudoaleatorias de la variable discreta seleccionada.
    
    Args:
        dist_name: Nombre de la distribución ('Poisson', 'Binomial', 'Binomial Negativa', 'Geométrica').
        params: Diccionario de parámetros estimados.
        size: Cantidad de datos a generar (por defecto 10,000).
        seed: Semilla aleatoria para reproducibilidad.
        
    Returns:
        Array NumPy de valores enteros simulados.
    """
    rng = np.random.default_rng(seed)
    
    if dist_name == 'Poisson':
        lam = float(params.get('lambda', 1.0))
        return rng.poisson(lam, size)
    elif dist_name == 'Binomial':
        n = int(params.get('n', 10))
        p = float(params.get('p', 0.5))
        return rng.binomial(n, p, size)
    elif dist_name == 'Binomial Negativa':
        r = float(params.get('r', 1.0))
        p = float(params.get('p', 0.5))
        return rng.negative_binomial(r, p, size)
    elif dist_name == 'Geométrica':
        p = float(params.get('p', 0.5))
        return rng.geometric(p, size)
    else:
        raise ValueError(f"Distribución discreta no soportada para generación: {dist_name}")


def generate_continuous(dist_name, params, size=10000, seed=42):
    """
    Genera observaciones pseudoaleatorias de la variable continua seleccionada.
    
    Args:
        dist_name: Nombre de la distribución ('Normal', 'Exponencial', 'Lognormal', 'Gamma', 'Weibull', 'Uniforme').
        params: Diccionario de parámetros estimados.
        size: Cantidad de datos a generar (por defecto 10,000).
        seed: Semilla aleatoria para reproducibilidad.
        
    Returns:
        Array NumPy de valores reales simulados.
    """
    rng = np.random.default_rng(seed)
    
    if dist_name == 'Normal':
        mu = float(params.get('mu', 0.0))
        sigma = float(params.get('sigma', 1.0))
        return rng.normal(mu, sigma, size)
    elif dist_name == 'Exponencial':
        scale = float(params.get('scale', 1.0))
        return rng.exponential(scale, size)
    elif dist_name == 'Lognormal':
        mu = float(params.get('mu', 0.0))
        sigma = float(params.get('sigma', 1.0))
        return rng.lognormal(mu, sigma, size)
    elif dist_name == 'Gamma':
        shape = float(params.get('shape', 1.0))
        scale = float(params.get('scale', 1.0))
        return rng.gamma(shape, scale, size)
    elif dist_name == 'Weibull':
        shape = float(params.get('shape', 1.0))
        scale = float(params.get('scale', 1.0))
        return scale * rng.weibull(shape, size)
    elif dist_name == 'Uniforme':
        a = float(params.get('a', 0.0))
        b = float(params.get('b', 1.0))
        return rng.uniform(a, b, size)
    else:
        raise ValueError(f"Distribución continua no soportada para generación: {dist_name}")


def get_method_description(dist_name):
    """Retorna información y fundamentación académica del método de generación."""
    if dist_name in DISCRETE_METHODS:
        info = DISCRETE_METHODS[dist_name]
        return {
            'method': info['method'],
            'description': info['academic_explanation']
        }
    elif dist_name in CONTINUOUS_METHODS:
        info = CONTINUOUS_METHODS[dist_name]
        return {
            'method': info['method'],
            'description': info['academic_explanation']
        }
    return {
        'method': 'Método numérico estándar',
        'description': 'Generación mediante algoritmos pseudoaleatorios de NumPy/SciPy.'
    }


def compare_distributions(real_data, simulated_data):
    """
    Compara rigurosamente la muestra empírica real con la muestra simulada:
    - Estadísticos descriptivos comparados lado a lado
    - Diferencia absoluta y porcentaje de discrepancia
    - Pruebas estadísticas formales de dos muestras (Kolmogorov-Smirnov y Mann-Whitney)
    
    Args:
        real_data: Array de datos reales observados.
        simulated_data: Array de datos simulados generados.
        
    Returns:
        Diccionario con tabla comparativa y pruebas estadísticas.
    """
    r = np.array(real_data, dtype=float)
    s = np.array(simulated_data, dtype=float)
    r = r[np.isfinite(r)]
    s = s[np.isfinite(s)]
    
    def calc_stats(arr):
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        return {
            'n': int(len(arr)),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            'var': float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0,
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'skewness': float(pd.Series(arr).skew()) if len(arr) > 2 else 0.0,
            'kurtosis': float(pd.Series(arr).kurtosis()) if len(arr) > 2 else 0.0,
            'percentiles': {f'p{p}': float(np.percentile(arr, p)) for p in percentiles}
        }
        
    real_stats = calc_stats(r)
    sim_stats = calc_stats(s)
    
    # Calcular errores relativos y diferencias absolutas
    diff = {
        'mean_diff': abs(real_stats['mean'] - sim_stats['mean']),
        'mean_rel_error_pct': abs(real_stats['mean'] - sim_stats['mean']) / abs(real_stats['mean']) * 100 if real_stats['mean'] != 0 else 0.0,
        'std_diff': abs(real_stats['std'] - sim_stats['std']),
        'std_rel_error_pct': abs(real_stats['std'] - sim_stats['std']) / abs(real_stats['std']) * 100 if real_stats['std'] != 0 else 0.0,
        'median_diff': abs(real_stats['median'] - sim_stats['median']),
    }
    
    # Pruebas estadísticas de dos muestras
    ks_res = ks_2samp_test(r, s)
    mw_res = mann_whitney_test(r, s)
    
    return {
        'real_stats': real_stats,
        'simulated_stats': sim_stats,
        'differences': diff,
        'ks_test': ks_res,
        'mann_whitney': mw_res
    }
