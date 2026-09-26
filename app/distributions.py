"""
Módulo de ajuste de distribuciones de probabilidad teóricas (Discretas y Continuas).
Incluye estimación de parámetros (MLE y método de momentos), cálculo de probabilidades
esperadas, generación de curvas teóricas para gráficos Plotly y pruebas de bondad de ajuste.
"""

import numpy as np
from scipy import stats
import math
from statistics import chi_square_test, ks_test


# =============================================================================
# DISTRIBUCIONES DISCRETAS
# =============================================================================

class DiscreteDistribution:
    """Clase base para distribuciones de probabilidad discretas."""
    
    def __init__(self, name):
        self.name = name
        self.params = {}
        self.valid = True
        self.error = None
        
    def fit(self, data):
        raise NotImplementedError
        
    def pmf(self, k):
        raise NotImplementedError
        
    def expected_freq(self, observed_values, observed_freq):
        raise NotImplementedError
        
    def description(self):
        raise NotImplementedError


class PoissonDistribution(DiscreteDistribution):
    """
    Distribución de Poisson:
    P(X = k) = (λ^k * e^(-λ)) / k!,  k = 0, 1, 2, ...
    Parámetro: λ (tasa promedio de ocurrencia).
    Propiedad teórica fundamental: Media = Varianza = λ (equidispersión).
    """
    
    def __init__(self):
        super().__init__("Poisson")
        
    def fit(self, data):
        try:
            mean_val = float(np.mean(data))
            var_val = float(np.var(data, ddof=1)) if len(data) > 1 else mean_val
            
            if mean_val < 0:
                self.valid = False
                self.error = "Poisson requiere datos no negativos (λ >= 0)."
                return
                
            self.params['lambda'] = max(0.0001, mean_val)
            self.params['dispersion_ratio'] = var_val / mean_val if mean_val > 0 else 1.0
            self.valid = True
        except Exception as e:
            self.valid = False
            self.error = str(e)
            
    def pmf(self, k):
        if not self.valid or k < 0:
            return 0.0
        lam = self.params.get('lambda', 1.0)
        return float(stats.poisson.pmf(int(k), lam))
        
    def expected_freq(self, observed_values, observed_freq):
        lam = self.params.get('lambda', 1.0)
        n_total = sum(observed_freq)
        expected = []
        for val in observed_values:
            try:
                p = stats.poisson.pmf(int(val), lam)
                expected.append(float(p * n_total))
            except Exception:
                expected.append(0.01)
        return expected
        
    def description(self):
        return (
            "Modela el número de eventos que ocurren en un intervalo o espacio fijo a una tasa constante λ. "
            "Asume independencia entre eventos y equidispersión teórica (Var(X) = E(X))."
        )


class BinomialDistribution(DiscreteDistribution):
    """
    Distribución Binomial:
    P(X = k) = C(n, k) * p^k * (1-p)^(n-k),  k = 0, 1, ..., n
    Parámetros: n (número de ensayos), p (probabilidad de éxito).
    Propiedad teórica fundamental: Varianza = n*p*(1-p) < Media = n*p (subdispersión).
    """
    
    def __init__(self):
        super().__init__("Binomial")
        
    def fit(self, data):
        try:
            mean_val = float(np.mean(data))
            var_val = float(np.var(data, ddof=1)) if len(data) > 1 else mean_val
            max_val = int(np.max(data))
            
            if min(data) < 0:
                self.valid = False
                self.error = "La distribución Binomial requiere valores no negativos."
                return
                
            # Método de momentos si var < mean
            if var_val < mean_val and mean_val > 0:
                p_est = 1.0 - (var_val / mean_val)
                n_est = max(max_val, int(round(mean_val / p_est)))
                p_est = min(max(p_est, 0.001), 0.999)
            else:
                # Si los datos están sobredispersos, usar n = max_val observado
                n_est = max(1, max_val)
                p_est = min(max(mean_val / n_est, 0.001), 0.999)
                
            self.params['n'] = int(n_est)
            self.params['p'] = float(p_est)
            self.valid = True
        except Exception as e:
            self.valid = False
            self.error = str(e)
            
    def pmf(self, k):
        if not self.valid or k < 0:
            return 0.0
        n = self.params.get('n', 10)
        p = self.params.get('p', 0.5)
        return float(stats.binom.pmf(int(k), n, p))
        
    def expected_freq(self, observed_values, observed_freq):
        n = self.params.get('n', 10)
        p = self.params.get('p', 0.5)
        n_total = sum(observed_freq)
        expected = []
        for val in observed_values:
            try:
                prob = stats.binom.pmf(int(val), n, p)
                expected.append(float(prob * n_total))
            except Exception:
                expected.append(0.01)
        return expected
        
    def description(self):
        return (
            "Modela el número de éxitos en n ensayos de Bernoulli independientes con probabilidad p constante. "
            "Acotada superiormente en n; asume subdispersión teórica (Var(X) < E(X))."
        )


class NegativeBinomialDistribution(DiscreteDistribution):
    """
    Distribución Binomial Negativa:
    Modela el número de fallos antes de observar r éxitos.
    Propiedad teórica fundamental: Varianza > Media (sobredispersión).
    Ideal para datos con aglomeración o varianza superior a Poisson.
    """
    
    def __init__(self):
        super().__init__("Binomial Negativa")
        
    def fit(self, data):
        try:
            mean_val = float(np.mean(data))
            var_val = float(np.var(data, ddof=1)) if len(data) > 1 else mean_val * 1.5
            
            if min(data) < 0:
                self.valid = False
                self.error = "La Binomial Negativa requiere valores no negativos."
                return
                
            # Método de momentos para sobredispersión (Var > Mean)
            if var_val > mean_val and mean_val > 0:
                p_est = mean_val / var_val
                r_est = (mean_val ** 2) / (var_val - mean_val)
            else:
                p_est = 0.5
                r_est = max(0.5, mean_val)
                
            self.params['r'] = float(max(0.05, r_est))
            self.params['p'] = float(min(max(p_est, 0.001), 0.999))
            self.valid = True
        except Exception as e:
            self.valid = False
            self.error = str(e)
            
    def pmf(self, k):
        if not self.valid or k < 0:
            return 0.0
        r = self.params.get('r', 1.0)
        p = self.params.get('p', 0.5)
        return float(stats.nbinom.pmf(int(k), r, p))
        
    def expected_freq(self, observed_values, observed_freq):
        r = self.params.get('r', 1.0)
        p = self.params.get('p', 0.5)
        n_total = sum(observed_freq)
        expected = []
        for val in observed_values:
            try:
                prob = stats.nbinom.pmf(int(val), r, p)
                expected.append(float(prob * n_total))
            except Exception:
                expected.append(0.01)
        return expected
        
    def description(self):
        return (
            "Modela conteos discretos con sobredispersión (Var(X) > E(X)). "
            "Generaliza a Poisson permitiendo heterogeneidad no observada en la tasa de eventos."
        )


class GeometricDistribution(DiscreteDistribution):
    """
    Distribución Geométrica:
    P(X = k) = (1-p)^(k-1) * p  (para k >= 1)
    Modela el número de ensayos necesarios hasta obtener el primer éxito.
    """
    
    def __init__(self):
        super().__init__("Geométrica")
        
    def fit(self, data):
        try:
            clean = [x for x in data if x > 0]
            if len(clean) == 0:
                self.valid = False
                self.error = "La distribución Geométrica requiere valores enteros positivos (k >= 1)."
                return
            mean_val = float(np.mean(clean))
            p_est = 1.0 / mean_val if mean_val >= 1.0 else 0.5
            self.params['p'] = float(min(max(p_est, 0.001), 0.999))
            self.valid = True
        except Exception as e:
            self.valid = False
            self.error = str(e)
            
    def pmf(self, k):
        if not self.valid or k < 1:
            return 0.0
        p = self.params.get('p', 0.5)
        return float(stats.geom.pmf(int(k), p))
        
    def expected_freq(self, observed_values, observed_freq):
        p = self.params.get('p', 0.5)
        n_total = sum(observed_freq)
        expected = []
        for val in observed_values:
            try:
                prob = stats.geom.pmf(int(val), p) if val >= 1 else 0.0
                expected.append(float(prob * n_total))
            except Exception:
                expected.append(0.01)
        return expected
        
    def description(self):
        return (
            "Modela el número de ensayos de Bernoulli independientes hasta la aparición del primer éxito. "
            "Posee la propiedad de pérdida de memoria en el dominio discreto."
        )


# =============================================================================
# DISTRIBUCIONES CONTINUAS
# =============================================================================

class ContinuousDistribution:
    """Clase base para distribuciones de probabilidad continuas."""
    
    def __init__(self, name):
        self.name = name
        self.params = {}
        self.valid = True
        self.error = None
        
    def fit(self, data):
        raise NotImplementedError
        
    def pdf(self, x):
        raise NotImplementedError
        
    def cdf(self, x):
        raise NotImplementedError
        
    def curve_points(self, x_min, x_max, n=200):
        """Genera puntos (x, y) de la función de densidad para graficar."""
        xs = np.linspace(x_min, x_max, n)
        ys = [float(self.pdf(x)) for x in xs]
        return xs.tolist(), ys
        
    def description(self):
        raise NotImplementedError


class NormalDistribution(ContinuousDistribution):
    """Distribución Normal (Gaussiana) N(μ, σ^2)."""
    
    def __init__(self):
        super().__init__("Normal")
        
    def fit(self, data):
        try:
            self.params['mu'] = float(np.mean(data))
            self.params['sigma'] = float(np.std(data, ddof=1)) if len(data) > 1 else 1.0
            self.valid = bool(self.params['sigma'] > 0)
        except Exception as e:
            self.valid = False
            self.error = str(e)
            
    def pdf(self, x):
        if not self.valid:
            return 0.0
        return float(stats.norm.pdf(x, self.params['mu'], self.params['sigma']))
        
    def cdf(self, x):
        if not self.valid:
            return 0.0
        return float(stats.norm.cdf(x, self.params['mu'], self.params['sigma']))
        
    def description(self):
        return (
            "Distribución simétrica en campana fundamentada en el Teorema del Límite Central. "
            "Adecuada para variables resultantes de la suma de múltiples efectos aleatorios independientes."
        )


class ExponentialDistribution(ContinuousDistribution):
    """Distribución Exponencial Exp(λ)."""
    
    def __init__(self):
        super().__init__("Exponencial")
        
    def fit(self, data):
        try:
            pos_data = data[data > 0]
            if len(pos_data) == 0:
                self.valid = False
                self.error = "Exponencial requiere valores positivos."
                return
            mean_val = float(np.mean(pos_data))
            self.params['lambda'] = float(1.0 / mean_val) if mean_val > 0 else 1.0
            self.params['scale'] = float(mean_val)
            self.valid = True
        except Exception as e:
            self.valid = False
            self.error = str(e)
            
    def pdf(self, x):
        if not self.valid or x < 0:
            return 0.0
        return float(stats.expon.pdf(x, scale=self.params['scale']))
        
    def cdf(self, x):
        if not self.valid or x < 0:
            return 0.0
        return float(stats.expon.cdf(x, scale=self.params['scale']))
        
    def description(self):
        return (
            "Modela el tiempo entre eventos consecutivos en un proceso de Poisson homogéneo. "
            "Caracterizada por tasa de falla constante y propiedad de pérdida de memoria."
        )


class LognormalDistribution(ContinuousDistribution):
    """Distribución Lognormal LogN(μ, σ^2)."""
    
    def __init__(self):
        super().__init__("Lognormal")
        
    def fit(self, data):
        try:
            pos_data = data[data > 0]
            if len(pos_data) < 5:
                self.valid = False
                self.error = "Lognormal requiere datos estrictamente positivos (x > 0)."
                return
            log_vals = np.log(pos_data)
            self.params['mu'] = float(np.mean(log_vals))
            self.params['sigma'] = float(np.std(log_vals, ddof=1))
            self.params['scale'] = float(np.exp(self.params['mu']))
            self.valid = bool(self.params['sigma'] > 0)
        except Exception as e:
            self.valid = False
            self.error = str(e)
            
    def pdf(self, x):
        if not self.valid or x <= 0:
            return 0.0
        return float(stats.lognorm.pdf(x, s=self.params['sigma'], scale=self.params['scale']))
        
    def cdf(self, x):
        if not self.valid or x <= 0:
            return 0.0
        return float(stats.lognorm.cdf(x, s=self.params['sigma'], scale=self.params['scale']))
        
    def description(self):
        return (
            "Variable cuyo logaritmo natural sigue una distribución normal. "
            "Apropiada para variables asimétricas positivas con cola derecha larga (tiempos de reacción, duraciones)."
        )


class GammaDistribution(ContinuousDistribution):
    """Distribución Gamma Gamma(α, β)."""
    
    def __init__(self):
        super().__init__("Gamma")
        
    def fit(self, data):
        try:
            pos_data = data[data > 0]
            if len(pos_data) < 5:
                self.valid = False
                self.error = "Gamma requiere valores positivos (x > 0)."
                return
            shape, loc, scale = stats.gamma.fit(pos_data, floc=0)
            self.params['shape'] = float(shape)
            self.params['scale'] = float(scale)
            self.valid = bool(shape > 0 and scale > 0)
        except Exception as e:
            self.valid = False
            self.error = str(e)
            
    def pdf(self, x):
        if not self.valid or x <= 0:
            return 0.0
        return float(stats.gamma.pdf(x, a=self.params['shape'], scale=self.params['scale']))
        
    def cdf(self, x):
        if not self.valid or x <= 0:
            return 0.0
        return float(stats.gamma.cdf(x, a=self.params['shape'], scale=self.params['scale']))
        
    def description(self):
        return (
            "Distribución flexible de dos parámetros que generaliza la exponencial y la Erlang. "
            "Modela tiempos de espera acumulados hasta α eventos en un proceso estocástico."
        )


class WeibullDistribution(ContinuousDistribution):
    """Distribución de Weibull Weibull(c, scale)."""
    
    def __init__(self):
        super().__init__("Weibull")
        
    def fit(self, data):
        try:
            pos_data = data[data > 0]
            if len(pos_data) < 5:
                self.valid = False
                self.error = "Weibull requiere datos positivos."
                return
            shape, loc, scale = stats.weibull_min.fit(pos_data, floc=0)
            self.params['shape'] = float(shape)
            self.params['scale'] = float(scale)
            self.valid = bool(shape > 0 and scale > 0)
        except Exception as e:
            self.valid = False
            self.error = str(e)
            
    def pdf(self, x):
        if not self.valid or x <= 0:
            return 0.0
        return float(stats.weibull_min.pdf(x, c=self.params['shape'], scale=self.params['scale']))
        
    def cdf(self, x):
        if not self.valid or x <= 0:
            return 0.0
        return float(stats.weibull_min.cdf(x, c=self.params['shape'], scale=self.params['scale']))
        
    def description(self):
        return (
            "Ampliamente utilizada en análisis de supervivencia y fiabilidad. "
            "Su tasa de fallo puede ser creciente, decreciente o constante según el parámetro de forma c."
        )


class UniformDistribution(ContinuousDistribution):
    """Distribución Uniforme continua U(a, b)."""
    
    def __init__(self):
        super().__init__("Uniforme")
        
    def fit(self, data):
        try:
            a = float(np.min(data))
            b = float(np.max(data))
            self.params['a'] = a
            self.params['b'] = b
            self.valid = bool(b > a)
        except Exception as e:
            self.valid = False
            self.error = str(e)
            
    def pdf(self, x):
        if not self.valid:
            return 0.0
        a, b = self.params['a'], self.params['b']
        return 1.0 / (b - a) if a <= x <= b else 0.0
        
    def cdf(self, x):
        if not self.valid:
            return 0.0
        a, b = self.params['a'], self.params['b']
        if x < a: return 0.0
        if x > b: return 1.0
        return (x - a) / (b - a)
        
    def description(self):
        return "Asigna densidad constante a todos los valores dentro del intervalo acotado [a, b]."


# =============================================================================
# FUNCIONES PRINCIPALES DE AJUSTE
# =============================================================================

def get_discrete_distributions():
    """Retorna las distribuciones discretas candidatas obligatorias."""
    return [
        PoissonDistribution(),
        BinomialDistribution(),
        NegativeBinomialDistribution(),
        GeometricDistribution(),
    ]


def get_continuous_distributions():
    """Retorna las distribuciones continuas candidatas obligatorias."""
    return [
        NormalDistribution(),
        ExponentialDistribution(),
        LognormalDistribution(),
        GammaDistribution(),
        WeibullDistribution(),
        UniformDistribution(),
    ]


def fit_discrete_distributions(data, observed_values, observed_freq):
    """
    Ajusta todas las distribuciones discretas candidatas y evalúa su bondad de ajuste con Chi-cuadrado.
    
    Args:
        data: Array con todas las observaciones discretas.
        observed_values: Lista de valores únicos ordenados.
        observed_freq: Lista de frecuencias observadas correspondientes.
        
    Returns:
        Lista de diccionarios con resultados de cada distribución.
    """
    results = []
    candidates = get_discrete_distributions()
    
    for dist in candidates:
        res = {
            'name': dist.name,
            'valid': False,
            'params': {},
            'chi2_stat': None,
            'p_value': None,
            'df': None,
            'expected_freq': [],
            'description': dist.description(),
            'error': None
        }
        
        try:
            dist.fit(data)
            res['valid'] = dist.valid
            res['params'] = dist.params
            
            if dist.valid:
                exp_freq = dist.expected_freq(observed_values, observed_freq)
                res['expected_freq'] = exp_freq
                
                # Número de parámetros estimados restados en g.l.
                ddof = len(dist.params)
                test_res = chi_square_test(observed_freq, exp_freq, ddof=ddof)
                
                res['chi2_stat'] = test_res.get('statistic')
                res['p_value'] = test_res.get('p_value')
                res['df'] = test_res.get('df')
            else:
                res['error'] = dist.error
        except Exception as e:
            res['valid'] = False
            res['error'] = str(e)
            
        results.append(res)
        
    return results


def fit_continuous_distributions(data):
    """
    Ajusta todas las distribuciones continuas candidatas y evalúa su bondad de ajuste con Kolmogorov-Smirnov.
    
    Args:
        data: Array con observaciones continuas.
        
    Returns:
        Lista de diccionarios con resultados y curvas teóricas.
    """
    results = []
    clean_data = np.array(data, dtype=float)
    clean_data = clean_data[np.isfinite(clean_data)]
    
    x_min = float(np.min(clean_data))
    x_max = float(np.max(clean_data))
    
    scipy_map = {
        'Normal': 'norm',
        'Exponencial': 'expon',
        'Lognormal': 'lognorm',
        'Gamma': 'gamma',
        'Weibull': 'weibull_min',
        'Uniforme': 'uniform'
    }
    
    for dist in get_continuous_distributions():
        res = {
            'name': dist.name,
            'valid': False,
            'params': {},
            'ks_stat': None,
            'p_value': None,
            'curve_x': [],
            'curve_y': [],
            'description': dist.description(),
            'error': None
        }
        
        try:
            dist.fit(clean_data)
            res['valid'] = dist.valid
            res['params'] = dist.params
            
            if dist.valid:
                scipy_name = scipy_map.get(dist.name, 'norm')
                ks_res = ks_test(clean_data, scipy_name, dist.params)
                res['ks_stat'] = ks_res.get('statistic')
                res['p_value'] = ks_res.get('p_value')
                
                # Curva teórica para superponer sobre el histograma en Plotly
                cx, cy = dist.curve_points(x_min, x_max, n=200)
                res['curve_x'] = cx
                res['curve_y'] = cy
            else:
                res['error'] = dist.error
        except Exception as e:
            res['valid'] = False
            res['error'] = str(e)
            
        results.append(res)
        
    return results
