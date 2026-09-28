"""
Módulo de ajuste de distribuciones de probabilidad teóricas (Discretas y Continuas).
Incluye estimación de parámetros (MLE y método de momentos), cálculo de probabilidades
esperadas, generación de curvas teóricas para gráficos Plotly y pruebas de bondad de ajuste.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

from statistics import chi_square_test, ks_test


# =============================================================================
# DISTRIBUCIONES DISCRETAS
# =============================================================================

class DiscreteDistribution:
    """Clase base para distribuciones de probabilidad discretas."""

    #: Número de parámetros libremente estimados por el ajuste (se resta en los g.l.)
    n_params = 1
    #: Soporte teórico como (límite inferior, límite superior); None = no acotado.
    support = (0.0, None)
    #: True si el ajuste usa directamente Statistics de la muestra (ajuste circular)
    circular_fit = False

    def __init__(self, name):
        self.name = name
        self.params = {}
        #: Estadísticos de diagnóstico de la muestra. No son parámetros del modelo,
        #: por lo que se guardan aparte de self.params para no reportarlos como
        #: si fueran estimados ni contaminar n_params.
        self.diagnostics = {}
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

    def scipy_mvsk(self):
        """
        (media, varianza, asimetria, curtosis-exceso) de la distribucion ajustada,
        tomados de la fuente analitica de SciPy.

        Se delega deliberadamente en SciPy en lugar de escribir las formulas
        cerradas a mano: las expresiones de los momentos de cuarto orden se
        confunden con facilidad (la curtosis es *exceso*, no Pearson, y los
        coeficientes de Weibull y Lognormal son faciles de equivocar), y un error
        en ellas falsearia el contraste de momentos y, con el, la seleccion
        multiobjetiva. Cada subclase implementa esta operacion.
        """
        raise NotImplementedError

    def theoretical_moments(self):
        """
        Momentos teoricos implied por los parametros estimados.

        Se usan para contrastar el modelo contra la muestra con independencia del
        estadistico de bondad de ajuste: una distribucion cuyo modelo de momentos
        no reproduce los momentos muestrales no puede ser la adecuada, por muy
        favorable que sea su estadistico.

        Returns:
            Diccionario con 'mean', 'var', 'skew' y 'kurt' (curtosis de exceso,
            misma convencion que pandas .kurt()), o {} si no aplica.
        """
        if not self.valid or not self.params:
            return {}
        try:
            mvsk = self.scipy_mvsk()
        except Exception:
            return {}
        if mvsk is None:
            return {}

        out = {}
        for key, value in zip(('mean', 'var', 'skew', 'kurt'), mvsk):
            try:
                out[key] = float(value) if value is not None and np.isfinite(value) else None
            except (TypeError, ValueError):
                out[key] = None
        return out

    def support_description(self):
        low, high = self.support
        if low is not None and high is None:
            return f"{{{_fmt_int(low)}, {1 if low == 0 else 2}, ...}} (no acotado superiormente)"
        if low is not None and high is not None:
            return f"{{{_fmt_int(low)}, ..., {_fmt_int(high)}}} (acotado)"
        return "no acotado"

    def support_bounds(self):
        """Límites efectivo del soporte teórico, usando los parámetros ya estimados."""
        low, high = self.support
        if high is None:
            high = math.inf
        if low is None:
            low = -math.inf
        return (low, high)


def _fmt_int(value):
    """Formatea un limite de soporte entero de forma legible."""
    return str(int(value))


class PoissonDistribution(DiscreteDistribution):
    """
    Distribución de Poisson:
    P(X = k) = (λ^k * e^(-λ)) / k!,  k = 0, 1, 2, ...
    Parámetro: λ (tasa promedio de ocurrencia).
    Propiedad teórica fundamental: Media = Varianza = λ (equidispersión).
    """
    
    n_params = 1
    support = (0.0, None)

    def __init__(self):
        super().__init__("Poisson")

    def scipy_mvsk(self):
        return stats.poisson.stats(self.params.get('lambda', 1.0), moments='mvsk')

    def fit(self, data):
        try:
            mean_val = float(np.mean(data))
            var_val = float(np.var(data, ddof=1)) if len(data) > 1 else mean_val
            
            if mean_val < 0:
                self.valid = False
                self.error = "Poisson requiere datos no negativos (λ >= 0)."
                return
                
            self.params['lambda'] = max(0.0001, mean_val)
            # La razón de dispersión es un DIAGNÓSTICO de la muestra, no un
            # parámetro del modelo: se guarda aparte para no reportarla como si
            # fuera un parámetro estimado (y para no contaminar n_params).
            self.diagnostics['dispersion_ratio'] = var_val / mean_val if mean_val > 0 else 1.0
            self.valid = True
        except Exception as e:
            self.valid = False
            self.error = str(e)
            
    def pmf(self, k):
        if not self.valid or k < 0:
            return 0.0
        lam = self.params.get('lambda', 1.0)
        return float(stats.poisson.pmf(int(k), lam))

    def cdf(self, x):
        if not self.valid:
            return 0.0
        lam = self.params.get('lambda', 1.0)
        if x < 0:
            return 0.0
        return float(stats.poisson.cdf(int(x), lam))

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

    n_params = 2
    support = (0.0, None)  # El límite superior real es n, fijado tras el ajuste

    def scipy_mvsk(self):
        return stats.binom.stats(self.params.get('n', 10), self.params.get('p', 0.5),
                                 moments='mvsk')

    def support_description(self):
        return f"{{0, ..., {_fmt_int(self.params.get('n', 0))}}} (acotado por n)"

    def support_bounds(self):
        return (0.0, float(self.params.get('n', 0)))

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

    def cdf(self, x):
        if not self.valid:
            return 0.0
        n = self.params.get('n', 10)
        p = self.params.get('p', 0.5)
        if x < 0:
            return 0.0
        return float(stats.binom.cdf(int(x), n, p))

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

    n_params = 2
    support = (0.0, None)

    def scipy_mvsk(self):
        return stats.nbinom.stats(self.params.get('r', 1.0), self.params.get('p', 0.5),
                                  moments='mvsk')

    def fit(self, data):
        try:
            mean_val = float(np.mean(data))
            var_val = float(np.var(data, ddof=1)) if len(data) > 1 else mean_val * 1.5
            
            if min(data) < 0:
                self.valid = False
                self.error = "La Binomial Negativa requiere valores no negativos."
                return

            if mean_val <= 0:
                self.valid = False
                self.error = "La Binomial Negativa requiere una media estrictamente positiva."
                return

            if var_val > mean_val:
                # Método de momentos para sobredispersión (Var > Media)
                p_est = mean_val / var_val
                r_est = (mean_val ** 2) / (var_val - mean_val)
            elif np.isclose(var_val, mean_val, rtol=1e-3):
                # Límite Poisson de la familia: p -> 1, r -> infinito
                p_est = 0.999
                r_est = mean_val * p_est / (1.0 - p_est)
            else:
                # La familia no puede reproducir subdispersión: Var(X) >= E(X) es
                # una propiedad estructural, no una cuestion de estimación.
                self.valid = False
                self.error = (
                    f"Inadmisible: la familia exige Var(X) >= E(X) y la muestra presenta "
                    f"subdispersión (Var = {var_val:.4f} < E = {mean_val:.4f}). "
                    "Ninguna elección de (r, p) puede reproducirla."
                )
                return

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

    def cdf(self, x):
        if not self.valid:
            return 0.0
        r = self.params.get('r', 1.0)
        p = self.params.get('p', 0.5)
        if x < 0:
            return 0.0
        return float(stats.nbinom.cdf(int(x), r, p))

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

    n_params = 1
    support = (1.0, None)

    def scipy_mvsk(self):
        return stats.geom.stats(self.params.get('p', 0.5), moments='mvsk')

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

    def cdf(self, x):
        if not self.valid:
            return 0.0
        p = self.params.get('p', 0.5)
        if x < 1:
            return 0.0
        return float(stats.geom.cdf(int(x), p))

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

    #: Número de parámetros libremente estimados por el ajuste
    n_params = 2
    #: Soporte teórico como (límite inferior, límite superior)
    support = (-math.inf, math.inf)
    #: True si el ajuste usa directamente el rango de la muestra (ajuste circular)
    circular_fit = False

    def __init__(self, name):
        self.name = name
        self.params = {}
        #: Estadísticos de diagnóstico de la muestra. No son parámetros del modelo,
        #: por lo que se guardan aparte de self.params para no reportarlos como
        #: si fueran estimados ni contaminar n_params.
        self.diagnostics = {}
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

    def scipy_mvsk(self):
        """
        (media, varianza, asimetria, curtosis-exceso) de la distribucion ajustada,
        tomados de la fuente analitica de SciPy. Ver DiscreteDistribution.scipy_mvsk.
        """
        raise NotImplementedError

    def theoretical_moments(self):
        """
        Momentos teoricos implied por los parametros MLE estimados.

        Se delega en la fuente analitica de SciPy mediante scipy_mvsk; cada
        subclase implementa esa operacion. Devuelve {} si el ajuste no es valido.
        """
        if not self.valid or not self.params:
            return {}
        try:
            mvsk = self.scipy_mvsk()
        except Exception:
            return {}
        if mvsk is None:
            return {}

        out = {}
        for key, value in zip(('mean', 'var', 'skew', 'kurt'), mvsk):
            try:
                out[key] = float(value) if value is not None and np.isfinite(value) else None
            except (TypeError, ValueError):
                out[key] = None
        return out

    def support_description(self):
        low, high = self.support
        if low == 0.0 and high == math.inf:
            return "(0, ∞) no acotada por la izquierda"
        if low == 0.0:
            return f"(0, {high:g})"
        if low == -math.inf and high == math.inf:
            return "(-∞, ∞) todo el real"
        return f"({low:g}, {high:g})"

    def support_bounds(self):
        """Límites efectivo del soporte teórico, usando los parámetros ya estimados."""
        low, high = self.support
        if high is None:
            high = math.inf
        if low is None:
            low = -math.inf
        return (low, high)


class NormalDistribution(ContinuousDistribution):
    """Distribución Normal (Gaussiana) N(μ, σ^2)."""

    support = (-math.inf, math.inf)

    def __init__(self):
        super().__init__("Normal")

    def scipy_mvsk(self):
        return stats.norm.stats(self.params.get('mu', 0.0), self.params.get('sigma', 1.0),
                                moments='mvsk')
        
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

    n_params = 1
    support = (0.0, math.inf)

    def __init__(self):
        super().__init__("Exponencial")

    def scipy_mvsk(self):
        return stats.expon.stats(scale=self.params.get('scale', 1.0), moments='mvsk')

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

    support = (0.0, math.inf)

    def __init__(self):
        super().__init__("Lognormal")

    def scipy_mvsk(self):
        return stats.lognorm.stats(self.params.get('sigma', 1.0),
                                   scale=self.params.get('scale', 1.0), moments='mvsk')

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

    support = (0.0, math.inf)

    def __init__(self):
        super().__init__("Gamma")

    def scipy_mvsk(self):
        return stats.gamma.stats(self.params.get('shape', 1.0),
                                scale=self.params.get('scale', 1.0), moments='mvsk')

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

    support = (0.0, math.inf)

    def __init__(self):
        super().__init__("Weibull")

    def scipy_mvsk(self):
        return stats.weibull_min.stats(self.params.get('shape', 1.0),
                                       scale=self.params.get('scale', 1.0), moments='mvsk')

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

    n_params = 2
    circular_fit = True  # a y b salen del rango de la muestra: el ajuste es circular

    def __init__(self):
        super().__init__("Uniforme")

    def support_description(self):
        return f"[{self.params.get('a', 0):.2f}, {self.params.get('b', 0):.2f}] (acotada)"

    def scipy_mvsk(self):
        a = float(self.params.get('a', 0.0))
        b = float(self.params.get('b', 1.0))
        return stats.uniform.stats(loc=a, scale=b - a, moments='mvsk')

    def support_bounds(self):
        return (float(self.params.get('a', 0.0)), float(self.params.get('b', 1.0)))
        
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
# SELECCIÓN MULTIOBJETIVO DE LA DISTRIBUCIÓN REPRESENTATIVA
# =============================================================================
#
# La nota "Importante" de la guía exige que la selección NO se base únicamente en
# una prueba estadística. Con muestras de decenas de miles de observaciones la
# potencia de χ² y KS es prácticamente 1 y todos los candidatos se rechazan con
# p ≈ 0, de modo que ordenar por p-valor o por el estadístico crudo no distingue
# nada. Se evalúan por eso tres criterios independientes y complementarios:
#
#   1. AJUSTE ESTADÍSTICO  (45 %): goodness-of-fit *relativo* entre candidatos.
#      Se usa χ²/g.l. o D de Kolmogorov-Smirnov normalizado contra el mejor
#      candidato, no el p-valor, que a esta escala no discrimina.
#
#   2. CONTRASTE DE MOMENTOS (30 %): error relativo entre los momentos
#      muestrales (media, varianza, asimetría y curtosis) y los momentos que
#      implica el modelo con los parámetros MLE. Es una verificación con
#      independencia del estadístico: una distribución cuyos momentos no
#      reproducen los de la muestra no puede ser la adecuada.
#
#   3. COHERENCIA DEL SOPORTE (25 %): compatibilidad entre el soporte teórico del
#      modelo y el soporte observado, que es donde la modelación se separa del
#      mero ajuste numérico. Una Normal ajustada a un tiempo de ronda asigna
#      masa de probabilidad a duraciones negativas, que el fenómeno prohíbe; una
#      Geométrica no puede tomar el valor 0, que el rango de habilidad sí exhibits.
#
# Los tres puntajes se normalizan a [0, 1] y se combinan con los pesos de
# SCORE_WEIGHTS. Todos los componentes se exponen en la interfaz y en el informe
# para que la decisión sea auditable y pueda ser revertida por el estudiante.

SCORE_WEIGHTS = {
    'gof': 0.45,
    'moment': 0.30,
    'support': 0.25,
}

#: Peso relativo de cada momento dentro del sub-puntaje de contraste.
MOMENT_WEIGHTS = {
    'mean': 1.0,
    'var': 1.0,
    'skew': 0.5,
    'kurt': 0.25,
}

#: Tolerancia de masa probabilística admisible fuera del rango observado.
#: Es un umbral de VIABILIDAD, no de puntaje: una familia que reparte más de este
#: 1 % de su masa en valores que el fenómeno prohíbe no es un modelo del fenómeno,
#: por muy bien que ajuste numéricamente. Ejemplo: una Normal ajustada a una
#: duración le asigna masa a tiempos negativos, que no existen.
VIABLE_MASS_THRESHOLD = 0.01


def sample_moment_profile(data):
    """Calcula los momentos muestrales usados como referencia del contraste."""
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {}

    series = pd.Series(arr)
    return {
        'n': int(arr.size),
        'mean': float(np.mean(arr)),
        'var': float(np.var(arr, ddof=1)),
        'std': float(np.std(arr, ddof=1)),
        'skew': float(series.skew()),
        'kurt': float(series.kurt()),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


def _relative_error(theory, sample, key):
    """Error relativo robusto entre un momento teórico y el muestral."""
    if theory is None or sample is None:
        return None
    if not np.isfinite(theory) or not np.isfinite(sample):
        return None
    denom = abs(sample)
    if denom < 1e-12:
        # Si el momento muestral es ~0 (p. ej. varianza nula) se compara en absoluto
        return abs(theory)
    return abs(theory - sample) / denom


def score_moment_contrast(sample_profile, theory_moments):
    """
    Compara los momentos del modelo con los de la muestra.

    Returns:
        (subscore en [0,1], detalle por momento)
    """
    if not sample_profile or not theory_moments:
        return 0.5, {}

    detail = {}
    weighted_error = 0.0
    weight_used = 0.0

    for key, weight in MOMENT_WEIGHTS.items():
        rel = _relative_error(theory_moments.get(key), sample_profile.get(key), key)
        if rel is None:
            continue
        detail[key] = {
            'sample': sample_profile.get(key),
            'theory': theory_moments.get(key),
            'rel_error': rel,
        }
        weighted_error += weight * rel
        weight_used += weight

    if weight_used == 0:
        return 0.5, detail

    mean_error = weighted_error / weight_used
    # Decaimiento suave: 0 % de error -> 1.0 ; 100 % de error -> 0.37 ; 300 % -> 0.14
    subscore = 1.0 / (1.0 + mean_error)
    return subscore, detail


def score_support_coherence(dist, sample_profile, data):
    """
    Evalúa la compatibilidad entre el modelo y las restricciones del fenómeno.

    Se aplican cuatro componentes de penalización con pesos distintos porque NO
    tienen el mismo significado estadístico:

    a) OBSERVACIONES FUERA DEL SOPORTE (dura): fracción de la muestra que la familia
       prohíbe. Incompatibilidad estructural, p. ej. una Geométrica no puede tomar el
       valor 0, que sí aparece en el rango de habilidad.

    b) MASA POR DEBAJO DEL SUELO ESTRUCTURAL (dura): probabilidad asignada a valores
       negativos. Un rango ni un tiempo de ronda existen por debajo de 0, así que
       cualquier masa ahí es un error del modelo, por bueno que sea su ajuste. Es
       el criterio que descalifica a la Normal como modelo de una duración.

    c) COLA SUPERIOR AL MÁXIMO OBSERVADO (blanda): en una muestra finita, toda
       familia no acotada deja algo de masa por encima del máximo observado. Es
       información sobre la cola, no un error, así que se penaliza levemente y
       NO invalida el modelo.

    d) ESCALA DEL LÍMITE SUPERIOR (blanda): si la familia es acotada pero su cota
       queda muy por encima del máximo observado, el ajuste exige un número de
       ensayos artificial para una variable acotada por diseño.

    Returns:
        (subscore en [0,1], detalle con los componentes y las incidencias)
    """
    if not sample_profile:
        return 0.5, {}

    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    xmin, xmax = sample_profile['min'], sample_profile['max']
    low, high = dist.support_bounds()

    issues = []

    # (a) Observaciones fuera del soporte teórico -> incompatibilidad dura
    if arr.size:
        below = float(np.mean(arr < low)) if np.isfinite(low) else 0.0
        above = float(np.mean(arr > high)) if np.isfinite(high) else 0.0
        violation_rate = below + above
    else:
        below = above = violation_rate = 0.0

    if violation_rate > 0:
        side = []
        if below > 0:
            side.append(f"{below * 100:.2f}% por debajo del mínimo teórico {low:g}")
        if above > 0:
            side.append(f"{above * 100:.2f}% por encima del máximo teórico {high:g}")
        issues.append("La familia prohíbe valores observados en la muestra: " + "; ".join(side) + ".")
    score_a = math.exp(-25.0 * violation_rate)

    # (b) Masa por debajo del suelo estructural del fenómeno (0 para rangos y duraciones)
    mass_below_floor = None
    score_b = 1.0
    try:
        mass_below_floor = max(0.0, float(dist.cdf(0.0)))
        score_b = math.exp(-20.0 * mass_below_floor)
        if mass_below_floor > VIABLE_MASS_THRESHOLD:
            issues.append(
                f"El modelo asigna {mass_below_floor * 100:.2f}% de su masa de probabilidad a "
                f"valores negativos, que el fenómeno prohíbe: {dist.name} no es un modelo "
                "viable para una variable no negativa, por favorable que sea su ajuste."
            )
    except Exception:
        mass_below_floor = None

    # (c) Cola por encima del máximo observado: señal sobre la cola, no error
    mass_above_max = None
    score_c = 1.0
    try:
        mass_above_max = max(0.0, 1.0 - float(dist.cdf(xmax)))
        score_c = max(0.0, 1.0 - mass_above_max)
    except Exception:
        mass_above_max = None

    # (d) Escala del límite superior en familias acotadas
    scale_ratio = None
    score_d = 1.0
    if np.isfinite(high) and high > 0 and xmax > 0:
        scale_ratio = high / xmax
        if scale_ratio > 1.0:
            score_d = max(0.0, 1.0 - min(1.0, math.log(scale_ratio) / math.log(50.0)))
            if scale_ratio > 1.5:
                issues.append(
                    f"El límite superior del modelo ({high:g}) excede {scale_ratio:.1f} veces el "
                    f"máximo observado ({xmax:g}): la cota es un artefacto del ajuste y no una "
                    "restricción del fenómeno."
                )

    subscore = score_a * score_b * score_c * score_d

    if dist.circular_fit:
        # El ajuste usa el rango de la propia muestra: la coherencia de soporte es
        # circular y no puede usarse como evidencia a favor del modelo.
        subscore = min(subscore, 0.5)
        issues.append("El ajuste es circular: los límites del soporte se leen del rango de la "
                      "muestra, de modo que la coincidencia del soporte no constituye evidencia.")

    # Criterio de VIABILIDAD contextual. Solo se consideran las incompatibilidades
    # ESTRUCTURALES (ningún valor observado prohibido y ninguna masa en valores
    # físicamente imposibles). La cola por encima del máximo observado NO invalida:
    # es esperable en cualquier muestra finita.
    viable = bool(violation_rate == 0.0
                  and (mass_below_floor is None or mass_below_floor <= VIABLE_MASS_THRESHOLD))

    detail = {
        'violation_rate': violation_rate,
        'mass_below_floor': mass_below_floor,
        'mass_above_sample_max': mass_above_max,
        'support_scale_ratio': scale_ratio,
        'theoretical_support': dist.support_description(),
        'components': {'observaciones_invalidas': score_a,
                       'masa_bajo_el_suelo': score_b,
                       'cola_sobre_el_maximo': score_c,
                       'escala_del_soporte': score_d},
        'viable': viable,
        'issues': issues,
    }
    return subscore, detail


def score_candidates(results, data, kind, fitted=None):
    """
    Anota cada resultado con el puntaje multiobjetivo y ordena los candidatos.

    Args:
        results: Lista de diccionarios devuelta por fit_*_distributions.
        data: Datos observados de la variable.
        kind: 'discreta' o 'continua'.
        fitted: Diccionario {nombre: instancia ya ajustada}. Es imprescindible que
            sean las instancias con los parámetros estimados, porque los momentos
            teóricos y los límites del soporte se leen de ellas.

    Returns:
        La misma lista, enriquecida con la clave 'score' en cada elemento.
    """
    profile = sample_moment_profile(data)

    gof_values = []
    for res in results:
        if not res.get('valid'):
            continue
        stat = res.get('chi2_stat') if kind == 'discreta' else res.get('ks_stat')
        if stat is None or not np.isfinite(stat):
            continue
        if kind == 'discreta':
            df = max(1, res.get('df') or 1)
            gof_values.append(abs(float(stat)) / df)
        else:
            gof_values.append(abs(float(stat)))
    best_gof = min(gof_values) if gof_values else None

    if fitted is None:
        fitted = {d.name: d for d in (get_discrete_distributions() if kind == 'discreta'
                                      else get_continuous_distributions())}

    for res in results:
        dist = fitted.get(res['name'])
        if dist is None or not res.get('valid'):
            res['score'] = None
            continue

        # --- 1. Ajuste estadístico relativo ---
        stat = res.get('chi2_stat') if kind == 'discreta' else res.get('ks_stat')
        if kind == 'discreta':
            df = max(1, res.get('df') or 1)
            gof = abs(float(stat)) / df if stat is not None and np.isfinite(stat) else None
        else:
            gof = abs(float(stat)) if stat is not None and np.isfinite(stat) else None
        if gof is None or best_gof is None or best_gof <= 0:
            gof_score = 0.5
        else:
            gof_score = min(1.0, best_gof / gof)

        # --- 2. Contraste de momentos ---
        theory = dist.theoretical_moments()
        moment_score, moment_detail = score_moment_contrast(profile, theory)

        # --- 3. Coherencia del soporte ---
        support_score, support_detail = score_support_coherence(dist, profile, data)

        total = (SCORE_WEIGHTS['gof'] * gof_score
                 + SCORE_WEIGHTS['moment'] * moment_score
                 + SCORE_WEIGHTS['support'] * support_score)

        res['score'] = {
            'total': float(total),
            'gof': float(gof_score),
            'gof_raw': float(gof) if gof is not None else None,
            'moment': float(moment_score),
            'moment_detail': moment_detail,
            'support': float(support_score),
            'support_detail': support_detail,
            'theory_moments': theory,
            'weights': dict(SCORE_WEIGHTS),
        }

    scored = [r for r in results if r.get('score')]
    if scored:
        scored.sort(key=lambda r: r['score']['total'], reverse=True)
        for rank, res in enumerate(scored, start=1):
            res['score']['rank'] = rank
    return results


def select_best(results, prefer_viable=True):
    """
    Devuelve el candidato más adecuado para el informe.

    Criterio de selección (nunca el p-valor): se ordena por puntaje multiobjetivo
    y, preferentemente, se restringe a las familias VIABLES, es decir aquellas cuyo
    soporte es compatible con el fenómeno. Se resuelve así el punto central de la
    guía: un buen ajuste estadístico no garantiza que la familia sea la adecuada
    para el contexto. Si ninguna familia resultara viable se cae al ranking completo
    para no dejar la página sin selección.
    """
    scored = [r for r in (results or []) if r.get('score')]
    if not scored:
        return None
    if prefer_viable:
        viable = [r for r in scored if r['score']['support_detail'].get('viable')]
        if viable:
            return max(viable, key=lambda r: r['score']['total'])
    return max(scored, key=lambda r: r['score']['total'])


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
    fitted = {}
    
    for dist in candidates:
        fitted[dist.name] = dist
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
            res['params'] = dict(dist.params)
            res['diagnostics'] = dict(dist.diagnostics)
            res['n_params'] = dist.n_params
            res['theoretical_moments'] = dist.theoretical_moments()
            res['support'] = dist.support_description()

            if dist.valid:
                exp_freq = dist.expected_freq(observed_values, observed_freq)
                res['expected_freq'] = exp_freq

                # Se restan únicamente los parámetros realmente estimados
                test_res = chi_square_test(observed_freq, exp_freq, ddof=dist.n_params)

                res['chi2_stat'] = test_res.get('statistic')
                res['p_value'] = test_res.get('p_value')
                res['df'] = test_res.get('df')
                res['chi2_bins'] = test_res.get('k_bins')
                res['chi2_message'] = test_res.get('message')
                res['tail_mass'] = test_res.get('tail_mass')
            else:
                res['error'] = dist.error
        except Exception as e:
            res['valid'] = False
            res['error'] = str(e)
            
        results.append(res)
        
    return score_candidates(results, data, 'discreta', fitted=fitted)


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
    fitted = {}
    
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
        fitted[dist.name] = dist
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
            res['params'] = dict(dist.params)
            res['diagnostics'] = dict(dist.diagnostics)
            res['n_params'] = dist.n_params
            res['theoretical_moments'] = dist.theoretical_moments()
            res['support'] = dist.support_description()

            if dist.valid:
                scipy_name = scipy_map.get(dist.name, 'norm')
                ks_res = ks_test(clean_data, scipy_name, dist.params)
                res['ks_stat'] = ks_res.get('statistic')
                res['p_value'] = ks_res.get('p_value')
                res['ks_error'] = ks_res.get('error')
                
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
        
    return score_candidates(results, clean_data, 'continua', fitted=fitted)
