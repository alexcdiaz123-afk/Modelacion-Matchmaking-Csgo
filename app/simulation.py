"""
Módulo de generación de variables aleatorias simuladas y comparación estadística.

Cada distribución candidata se genera con el algoritmo de transformación estudiado
en el curso, implementado explícitamente aquí: transformada inversa, Box-Muller,
Knuth, Marsaglia-Tsang y la mezcla Poisson-Gamma. Esto permite que el informe
académico describa fielmente el método empleado y que la evidencia gráfica y
estadística provenga del algoritmo implementado por el grupo.

Excepción declarada y deliberada: `poisson_knuth` y `binomial_bernoulli` son
algoritmos exactos pero de costo O(lambda) y O(n) por draw, respectivamente. Cuando
lambda >= 30 o n*p >= 30 su costo deja de ser razonable y se recurre al algoritmo
BTPE de `numpy`, que también es exacto. El método efectivamente usado se reporta
en la interfaz, de modo que la afirmación nunca es más fuerte que el código.

El motor de números pseudoaleatorios base es PCG64 (Permuted Congruential
Generator de 64 bits) a través de `numpy.random.Generator`, sobre el que se
aplican las transformaciones inversas, Box-Muller, Knuth y Marsaglia-Tsang.
"""

import math

import numpy as np
import pandas as pd
from statistics import ks_2samp_test, mann_whitney_test

#: Por encima de este lambda el algoritmo de Knuth deja de ser razonable (O(lambda)
#: draws por valor) y se usa BTPE de numpy, que también es exacto.
_KNUTH_MAX_LAMBDA = 30.0
#: Análogo para la suma de Bernoulli: a partir de n*p >= 30 conviene BTPE.
_BINOMIAL_MAX_NP = 30.0


# =============================================================================
# TRANSFORMACIONES INVERSAS Y ALGORITMOS DE GENERACIÓN
# =============================================================================

def _uniform_open(rng, size):
    """Uniformes en el intervalo abierto (0, 1), evitando log(0) = -inf."""
    return rng.random(size)


def poisson_knuth(rng, lam, size):
    """
    Algoritmo de Knuth (1969) para la distribución de Poisson.

    Genera U_1, U_2, ... i.i.d. U(0,1) y calcula el producto mientras
    Π U_i > e^(-λ). El número de factores consumidos menos uno es el valor de X.

        P(X = k) = e^(-λ) λ^k / k!  =  e^(-λ) (λ-1)! / k!  (si k >= 1)

    Acepta `lam` escalar o vectorial: la mezcla Poisson-Gamma necesita un rate
    distinto por cada observación, y el umbral e^(-λ) es entonces específico de
    cada elemento. Cuando `lam` es vectorial se particiona, aplicando Knuth a los
    elementos con λ < 30 y BTPE de `numpy` a los mayores, porque el costo O(λ) por
    draw dejaría de ser razonable.

    Returns:
        Array de enteros con soporte {0, 1, 2, ...}.
    """
    lam_arr = np.asarray(lam, dtype=float)

    if lam_arr.ndim == 0:
        lam_val = float(lam_arr)
        if lam_val <= 0:
            return np.zeros(size, dtype=np.int64)
        if lam_val >= _KNUTH_MAX_LAMBDA:
            return rng.poisson(lam_val, size)
        out = np.empty(size, dtype=np.int64)
        pending = np.ones(size, dtype=bool)
        sub_lam = np.full(size, lam_val, dtype=float)
    else:
        lam_arr = np.broadcast_to(lam_arr, (size,))
        pending = lam_arr < _KNUTH_MAX_LAMBDA
        out = np.zeros(size, dtype=np.int64)
        if pending.any():
            out[~pending] = rng.poisson(lam_arr[~pending])
        sub_lam = lam_arr[pending]

    if not pending.any():
        return out

    idx = np.flatnonzero(pending)
    threshold = np.exp(-sub_lam)
    k = np.zeros(idx.size, dtype=np.int64)
    active = np.arange(idx.size)
    products = np.ones(idx.size, dtype=float)

    while active.size:
        k[active] += 1
        products[active] *= rng.random(active.size)
        active = active[products[active] > threshold[active]]

    out[idx] = k - 1
    return out


def binomial_bernoulli(rng, n, p, size):
    """
    Binomial por suma de variables de Bernoulli independientes.

        P(X = k) = C(n, k) p^k (1-p)^(n-k)

    Cada X es el número de éxitos en n ensayos de Bernoulli. Es exacto siempre,
    pero su costo es O(n) por draw: se usa cuando n*p < 30 y se recurre a BTPE
    cuando el número de ensayos es grande.
    """
    n = int(n)
    if n <= 0:
        return np.zeros(size, dtype=np.int64)
    if n * p >= _BINOMIAL_MAX_NP:
        return rng.binomial(n, p, size)

    uniforms = rng.random((size, n))
    return (uniforms < p).sum(axis=1).astype(np.int64)


def negative_binomial_poisson_gamma(rng, r, p, size):
    """
    Binomial Negativa como mezcla Poisson-Gamma.

    Se simula primero el rate Y ~ Gamma(r, escala = (1-p)/p) con el algoritmo de
    Marsaglia-Tsang implementado en este módulo y luego X ~ Poisson(Y) con el
    algoritmo de Knuth también propio. Esta construcción es exacta y explica la
    sobredispersión característica: Var(X) = E(X) + E(X)^2/r, de modo que
    Var(X) > E(X) para todo r finito.
    """
    rate = gamma_marsaglia_tsang(rng, shape=r, scale=(1.0 - p) / p, size=size)
    return poisson_knuth(rng, rate, size)


def geometric_inverse(rng, p, size):
    """
    Geométrica por transformada inversa analítica.

    F(x) = 1 - (1-p)^x  =>  F^(-1)(u) = ln(1-u) / ln(1-p)

    Aplicando el teorema de la transformada inversa con U ~ U(0,1) se obtiene
    directamente el soporte entero {1, 2, 3, ...}.
    """
    u = _uniform_open(rng, size)
    draws = np.ceil(np.log1p(-u) / math.log1p(-p))
    return np.maximum(draws, 1).astype(np.int64)


def normal_box_muller(rng, mu, sigma, size):
    """
    Normal por transformación de Box-Muller (1958).

    A partir de un par de uniformes independientes U_1, U_2 ~ U(0,1) se
    construyen dos normales estándar independientes:

        Z_0 = sqrt(-2 ln U_1) cos(2π U_2)
        Z_1 = sqrt(-2 ln U_1) sen(2π U_2)

    y luego se aplica la transformación lineal X = μ + σ Z.

    Solo se devuelve Z_0: el par de normales que produce cada par de uniformes se
    descarta, de modo que el algoritmo consume dos uniformes por salida. Es una
    pérdida de eficiencia aceptable y deliberada, a cambio de un generador
    vectorizado sin estado, que es lo que necesitan las simulaciones de gran tamaño
    y el uso de semillas reproducibles.
    """
    u1 = _uniform_open(rng, size)
    u2 = rng.random(size)
    radius = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    return mu + sigma * (radius * np.cos(theta))


def lognormal_box_muller(rng, mu, sigma, size):
    """
    Lognormal por Box-Muller + transformación exponencial.

    Si Y ~ Normal(μ, σ²), entonces X = e^Y ~ Lognormal(μ, σ²). La componente
    normal se genera con Box-Muller y se eleva al exponente componente a
    componente.
    """
    z = normal_box_muller(rng, mu, sigma, size)
    return np.exp(z)


def exponential_inverse(rng, scale, size):
    """
    Exponencial por transformada inversa analítica.

    F(x) = 1 - e^(-x/β)  =>  F^(-1)(u) = -β ln(1-u)

    Como 1-U ~ U(0,1) cuando U ~ U(0,1), equivale a X = -β ln U.
    """
    u = _uniform_open(rng, size)
    return -scale * np.log1p(-u)


def weibull_inverse(rng, shape, scale, size):
    """
    Weibull por transformada inversa analítica.

    F(x) = 1 - exp(-(x/β)^c)  =>  F^(-1)(u) = β (-ln(1-u))^(1/c)
    """
    u = _uniform_open(rng, size)
    return scale * np.power(-np.log1p(-u), 1.0 / shape)


def uniform_affine(rng, a, b, size):
    """
    Uniforme por transformación lineal afín sobre el PCG64.

    X = a + (b - a) U,  con U ~ U(0,1) proveniente del generador congruencial.
    """
    return a + (b - a) * rng.random(size)


def gamma_marsaglia_tsang(rng, shape, scale, size):
    """
    Gamma por el método de aceptación-rechazo de Marsaglia y Tsang (2000).

    Para shape >= 1 se transforma una normal estándar en una Gamma mediante
    v = (1 + x/(3√d))³ con d = shape - 1/3, aceptando con probabilidad
    v · exp(d(1 - v + ln v) - x²/2). Para shape < 1 se aplica la propiedad de
    descomposición: si X ~ Gamma(shape + 1) entonces X · U^(1/shape) ~ Gamma(shape).
    """
    if shape <= 0 or scale <= 0:
        return np.zeros(size, dtype=float)

    if shape < 1.0:
        boosted = gamma_marsaglia_tsang(rng, shape + 1.0, 1.0, size)
        return boosted * np.power(_uniform_open(rng, size), 1.0 / shape) * scale

    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    out = np.empty(size, dtype=float)
    filled = 0

    while filled < size:
        batch = max(64, (size - filled) * 2)
        x = rng.standard_normal(batch)
        v = 1.0 + c * x
        valid = v > 0.0
        v = np.where(valid, v, 1.0) ** 3
        u = rng.random(batch)

        accept = valid & (
            (u < 1.0 - 0.0331 * x ** 4)
            | (np.log(u) < 0.5 * x ** 2 + d * (1.0 - v + np.log(v)))
        )
        accepted = d * v[accept]
        take = min(len(accepted), size - filled)
        out[filled:filled + take] = accepted[:take]
        filled += take

    return out * scale


# =============================================================================
# DOCUMENTACIÓN DE LOS MÉTODOS DE GENERACIÓN
# =============================================================================

_BASE_RNG_NOTE = (
    "El motor base es el Generador PCG64 de 64 bits (Permuted Congruential Generator) "
    "expuesto por NumPy como numpy.random.Generator, inicializado con la semilla "
    "configurable para garantizar reproducibilidad exacta de las corridas."
)

DISCRETE_METHODS = {
    'Poisson': {
        'code': 'knuth',
        'method': 'Algoritmo de Knuth (producto de uniformes)',
        'academic_explanation': (
            "Se generan variables uniformes independientes U_1, U_2, ... y se calcula el "
            "producto acumulado mientras Π U_i > e^(-λ). El número de factores consumidos "
            "menos uno entrega directamente el valor de la variable aleatoria, porque la "
            "función generadora de la Poisson es e^{λ(z-1)} y su desarrollo por factoriales "
            "coincide con la serie del producto de uniformes. El método es exacto y verificable "
            "a mano, pero su costo es O(λ) por número generado, por lo que sólo se emplea con "
            f"λ < 30 (aquí λ = λ̂ de la muestra). " + _BASE_RNG_NOTE
        ),
        'generator': lambda rng, params, size: poisson_knuth(rng, float(params.get('lambda', 1.0)), size),
    },
    'Binomial': {
        'code': 'bernoulli',
        'method': 'Suma de variables de Bernoulli independientes',
        'academic_explanation': (
            "Se realizan n ensayos independientes de Bernoulli(p) y se cuenta el número de "
            "exitos: X = Σ B_i, donde cada B_i se genera comparando un uniforme U(0,1) contra p. "
            "Por la propiedad de aditividad de la Binomial, el resultado sigue exactamente "
            "Binomial(n, p) con C(n, k) p^k (1-p)^(n-k). Es exacto pero cuesta O(n) por draw, "
            "por lo que con n·p >= 30 se recurre al algoritmo BTPE de NumPy. " + _BASE_RNG_NOTE
        ),
        'generator': lambda rng, params, size: binomial_bernoulli(
            rng, int(params.get('n', 10)), float(params.get('p', 0.5)), size),
    },
    'Binomial Negativa': {
        'code': 'poisson_gamma',
        'method': 'Mezcla Poisson-Gamma',
        'academic_explanation': (
            "Se fundamenta en la representación de la Binomial Negativa como una mezcla de "
            "distribuciones de Poisson con tasa aleatoria: se simula primero el rate "
            "Y ~ Gamma(r, escala = (1-p)/p) y luego X ~ Poisson(Y). La construcción es exacta y "
            "reproduce analíticamente la sobredispersión, pues Var(X) = E(X) + E(X)²/r > E(X). "
            + _BASE_RNG_NOTE
        ),
        'generator': lambda rng, params, size: negative_binomial_poisson_gamma(
            rng, float(params.get('r', 1.0)), float(params.get('p', 0.5)), size),
    },
    'Geométrica': {
        'code': 'inversa',
        'method': 'Transformada inversa analítica',
        'academic_explanation': (
            "Se aplica el teorema de la transformada inversa sobre F(x) = 1 - (1-p)^x, cuya "
            "inversa analítica es F^(-1)(u) = ln(1-u)/ln(1-p). Generando U ~ U(0,1) y evaluando "
            "X = ⌈ln(1-U)/ln(1-p)⌉ se obtiene directamente el soporte entero {1, 2, ...}. "
            "Es un método exacto, de costo O(1) por valor y sin rechazos. " + _BASE_RNG_NOTE
        ),
        'generator': lambda rng, params, size: geometric_inverse(rng, float(params.get('p', 0.5)), size),
    },
}

CONTINUOUS_METHODS = {
    'Normal': {
        'code': 'box_muller',
        'method': 'Transformación de Box-Muller',
        'academic_explanation': (
            "A partir de un par de uniformes independientes U_1, U_2 ~ U(0,1) se construyen dos "
            "normales estándar independientes mediante Z_0 = √(-2 ln U_1) · cos(2π U_2) y "
            "Z_1 = √(-2 ln U_1) · sen(2π U_2), y se aplica la transformación lineal X = μ + σ Z. "
            "El método es exacto (no hay rechazos) y verificable analíticamente; su única "
            "desventaja es el consumo de dos uniformes y la función trigonométrica por par. "
            + _BASE_RNG_NOTE
        ),
        'generator': lambda rng, params, size: normal_box_muller(
            rng, float(params.get('mu', 0.0)), float(params.get('sigma', 1.0)), size),
    },
    'Exponencial': {
        'code': 'inversa',
        'method': 'Transformada inversa analítica',
        'academic_explanation': (
            "La acumulada F(x) = 1 - e^(-x/β) tiene inversa analítica F^(-1)(u) = -β ln(1-u). "
            "Como 1-U también es uniforme en (0,1) cuando U lo es, la expresión se simplifica a "
            "X = -β ln(U). Es exacta, de costo O(1) y sin rechazos, y satisface la propiedad de "
            "pérdida de memoria propia de los procesos de Poisson homogéneos. " + _BASE_RNG_NOTE
        ),
        'generator': lambda rng, params, size: exponential_inverse(
            rng, float(params.get('scale', 1.0)), size),
    },
    'Lognormal': {
        'code': 'box_muller',
        'method': 'Box-Muller seguido de transformación exponencial',
        'academic_explanation': (
            "Por definición, si Y ~ Normal(μ, σ²) entonces X = e^Y sigue una distribución "
            "Lognormal(μ, σ²). Se genera la componente normal con Box-Muller y se aplica la "
            "función exponencial componente a componente. La transformación garantiza que el "
            "soporte sea (0, ∞) y que la moda, mediana y media queden ordenadas como "
            "e^{μ-σ²} < e^μ < e^{μ+σ²/2}. " + _BASE_RNG_NOTE
        ),
        'generator': lambda rng, params, size: lognormal_box_muller(
            rng, float(params.get('mu', 0.0)), float(params.get('sigma', 1.0)), size),
    },
    'Gamma': {
        'code': 'marsaglia_tsang',
        'method': 'Aceptación-rechazo de Marsaglia y Tsang',
        'academic_explanation': (
            "Para shape >= 1 se transforma una normal estándar en una Gamma mediante "
            "v = (1 + x/(3√d))³ con d = shape - 1/3, y se acepta el valor d·v con probabilidad "
            "v · exp(d(1 - v + ln v) - x²/2); el tiempo de aceptación esperado es 1. "
            "Para shape < 1 se aplica la propiedad de descomposición: si X ~ Gamma(shape + 1) "
            "entonces X · U^(1/shape) ~ Gamma(shape), lo que resuelve el caso no entero. "
            + _BASE_RNG_NOTE
        ),
        'generator': lambda rng, params, size: gamma_marsaglia_tsang(
            rng, float(params.get('shape', 1.0)), float(params.get('scale', 1.0)), size),
    },
    'Weibull': {
        'code': 'inversa',
        'method': 'Transformada inversa analítica',
        'academic_explanation': (
            "La acumulada F(x) = 1 - exp(-(x/β)^c) se invierte analíticamente en "
            "F^(-1)(u) = β·(-ln(1-u))^(1/c). Igualando u con un uniforme U ~ U(0,1) se obtiene "
            "directamente X = β(-ln U)^(1/c), un método exacto de costo O(1) sin rechazos. "
            "El parámetro de forma c gobierna la tasa de fallo: c < 1 decreciente, c = 1 "
            "exponencial y c > 1 creciente. " + _BASE_RNG_NOTE
        ),
        'generator': lambda rng, params, size: weibull_inverse(
            rng, float(params.get('shape', 1.0)), float(params.get('scale', 1.0)), size),
    },
    'Uniforme': {
        'code': 'lineal',
        'method': 'Transformación lineal afín sobre el PCG64',
        'academic_explanation': (
            "El generador congruencial PCG64 produce enteros uniformes de 64 bits que se "
            "dividen entre 2^64 para mapear al intervalo continuo [0, 1). Sobre ese uniforme "
            "se aplica la transformación lineal afín X = a + (b - a)·U, que es exacta porque "
            "es una aplicación monótona de un uniforme sobre el intervalo acotado. "
            + _BASE_RNG_NOTE
        ),
        'generator': lambda rng, params, size: uniform_affine(
            rng, float(params.get('a', 0.0)), float(params.get('b', 1.0)), size),
    },
}

ALL_METHODS = {**DISCRETE_METHODS, **CONTINUOUS_METHODS}


def generate_discrete(dist_name, params, size=10000, seed=42):
    """
    Genera observaciones pseudoaleatorias de la variable discreta seleccionada
    usando el algoritmo de transformación implementado en este módulo.

    Args:
        dist_name: Nombre de la distribución ('Poisson', 'Binomial', 'Binomial Negativa', 'Geométrica').
        params: Diccionario de parámetros estimados.
        size: Cantidad de datos a generar (por defecto 10,000).
        seed: Semilla aleatoria para reproducibilidad.

    Returns:
        Array NumPy de valores enteros simulados.
    """
    if dist_name not in DISCRETE_METHODS:
        raise ValueError(f"Distribución discreta no soportada para generación: {dist_name}")
    rng = np.random.default_rng(seed)
    return np.asarray(DISCRETE_METHODS[dist_name]['generator'](rng, params, int(size)))


def generate_continuous(dist_name, params, size=10000, seed=42):
    """
    Genera observaciones pseudoaleatorias de la variable continua seleccionada
    usando el algoritmo de transformación implementado en este módulo.

    Args:
        dist_name: Nombre de la distribución ('Normal', 'Exponencial', 'Lognormal', 'Gamma', 'Weibull', 'Uniforme').
        params: Diccionario de parámetros estimados.
        size: Cantidad de datos a generar (por defecto 10,000).
        seed: Semilla aleatoria para reproducibilidad.

    Returns:
        Array NumPy de valores reales simulados.
    """
    if dist_name not in CONTINUOUS_METHODS:
        raise ValueError(f"Distribución continua no soportada para generación: {dist_name}")
    rng = np.random.default_rng(seed)
    return np.asarray(CONTINUOUS_METHODS[dist_name]['generator'](rng, params, int(size)))


def get_method_description(dist_name):
    """Retorna información y fundamentación académica del método de generación."""
    info = ALL_METHODS.get(dist_name)
    if info is None:
        return {
            'method': 'Método numérico estándar',
            'description': 'Generación mediante algoritmos pseudoaleatorios de NumPy/SciPy.',
            'code': 'desconocido',
        }
    return {
        'method': info['method'],
        'description': info['academic_explanation'],
        'code': info['code'],
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
