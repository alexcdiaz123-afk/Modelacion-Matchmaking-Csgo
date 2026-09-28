# Simulador Estadístico de Matchmaking — CS:GO

Dashboard interactivo y plataforma académica universitaria para la modelación probabilística de variables aleatorias discretas y continuas a partir de datos reales de partidas competitivas de **Counter-Strike: Global Offensive (CS:GO)**.

---

## 🎯 Contexto del Fenómeno Modelado

El fenómeno estudiado corresponde al sistema de **Matchmaking (emparejamiento)** de un videojuego competitivo multijugador:

> *"Un jugador solicita una partida y entra en una cola estocástica. El sistema busca jugadores compatibles evaluando sus características estadísticas. Cuando se reúne la cantidad necesaria de participantes (10 jugadores para un encuentro competitivo 5v5), se crea la partida."*

El proyecto utiliza datos reales para:
1. **Identificar y modelar una variable aleatoria DISCRETA** (Rango de Habilidad del Atacante `att_rank`, observado entre 0 y 18; alternativamente `round` o `ct_alive`).
2. **Identificar y modelar una variable aleatoria CONTINUA** (`seconds`, el reloj de la ronda; alternativamente `ct_eq_val`, `t_eq_val`, `avg_match_rank` o los daños `hp_dmg`/`arm_dmg`). El dataset **no** trae una columna de duración de ronda: `seconds` es un reloj correlacionado entre rondas de una misma partida, no una duración independiente, y el informe lo declara como limitación explícita.
3. **Generar variables aleatorias simuladas** a partir de las distribuciones ajustadas mediante métodos computacionales rigurosos.
4. **Alimentar un simulador de cola de matchmaking** que aplica un algoritmo de compatibilidad por distancias normalizadas.

---

## 🔬 Flujo Metodológico Académico

```
DATOS REALES
    ↓
LIMPIEZA DE DATOS (DataFrame aislado)
    ↓
ANÁLISIS EXPLORATORIO (Momentos, percentiles, ECDF, KDE, Boxplot)
    ↓
SELECCIÓN DE VARIABLE DISCRETA & CONTINUA (Análisis de cardinalidad)
    ↓
DISTRIBUCIONES CANDIDATAS (Poisson, Binomial, NegBinomial, Geométrica / Normal, Exponencial, Lognormal, Gamma, Weibull)
    ↓
AJUSTE DE DISTRIBUCIONES (Máxima Verosimilitud; momentos teóricos analíticos de SciPy)
    ↓
PRUEBAS ESTADÍSTICAS (Chi-cuadrado χ², Kolmogorov-Smirnov, Shapiro-Wilk, Anderson-Darling, D'Agostino)
    ↓
SELECCIÓN MULTIOBJETIVO (Ajuste 45% + Momentos 30% + Soporte/Contexto 25%, restringida a familias VIABLES)
    ↓
GENERACIÓN PSEUDOALEATORIA (Knuth, Marsaglia-Tsang, Box-Muller, Transformada Inversa; BTPE solo como respaldo)
    ↓
COMPARACIÓN REAL VS SIMULADA (Pruebas de dos muestras propio y Mann-Whitney U)
    ↓
APLICACIÓN AL MATCHMAKING (Cola estocástica, algoritmo de distancia euclidiana y creación de partidas)
    ↓
INFORME ACADÉMICO AUTOMÁTICO (20 Secciones completas con datos reales)
```

---

## 🧭 Criterio de selección (punto central del ejercicio)

La familia elegida **nunca** se decide por el p-valor ni por el menor estadístico, porque con
N = 50 000 la prueba χ² tiene potencia casi 1 y rechaza a todas las candidatas con p ≈ 0, de modo
que el p-valor no discrimina. Se combinan tres criterios normalizados:

| Criterio | Peso | Qué mide |
|---|---|---|
| Ajuste (GOF) | 0.45 | χ² para discretas, KS para continuas |
| Momentos | 0.30 | Contraste de media, varianza, asimetría y curtosis |
| Soporte y contexto | 0.25 | Coherencia del soporte con el significado de la variable |

Antes del score se aplica un **filtro de viabilidad contextual**: una familia es inadmisible si su
soporte prohíbe algún valor observado o si asigna más del 1 % de su masa a valores imposibles
(en la práctica, `P(X < 0)` para una duración). Si ninguna sobrevive, se degrada al ranking completo.

Esto produce el resultado central del ejercicio: para `seconds`, la **Normal gana el ajuste
estadístico** (D = 0.0577, el menor de la tabla) pero es **NO VIABLE** porque asigna 3.98 % de su
masa a tiempos negativos. La **Weibull** es la mejor familia viable y es la que se reporta. Para
`att_rank`, gana la **Poisson**, mientras la Binomial Negativa queda inadmisible por estructura
(exige `Var(X) ≥ E(X)` y la muestra es subdispersa, con razón 0.9626) y la Geométrica porque su
soporte `{1, 2, ...}` excluye el 0 observado.

---

## 🛠️ Tecnologías Empleadas

- **Backend:** Python 3.9+, Flask, pandas, NumPy, SciPy.
- **Frontend:** HTML5, CSS3 personalizado (tema oscuro táctico inspirado en esports competitivos), JavaScript (ES6+), Bootstrap 5.
- **Visualizaciones interactivas:** Plotly.js.
- **Adquisición del dataset:** KaggleHub.

---

## 📁 Arquitectura del Proyecto

```
/Modelacion-Matchmaking csgo
├── app/
│   ├── app.py              # Controlador Flask principal y rutas API
│   ├── data_loader.py      # Detección de KaggleHub, carga optimizada y sugerencias
│   ├── preprocessing.py    # Limpieza sobre copias aisladas (duplicados, nulos, outliers)
│   ├── distributions.py    # Ajuste de distribuciones discretas y continuas (MLE)
│   ├── simulation.py       # Generación de variables (10,000 obs) y comparación 2-muestras
│   ├── statistics.py       # Pruebas estadísticas (χ², KS, Shapiro, Anderson, Q-Q Plot)
│   ├── stats.py            # Puente de compatibilidad hacia statistics.py
│   ├── matchmaking.py      # Simulador de cola estocástica y algoritmo de compatibilidad
│   ├── utils.py            # Estadísticas descriptivas, frecuencias, KDE, ECDF y codificador JSON
│   ├── templates/          # Plantillas HTML con Bootstrap 5 y Plotly.js
│   │   ├── base.html       # Estructura base con barra lateral de 9 secciones
│   │   ├── index.html      # Módulo 1: Contexto y Dashboard
│   │   ├── dataset.html    # Módulo 2: Explorador multivariable del Dataset
│   │   ├── cleaning.html   # Módulo 3: Limpieza y resumen Antes/Después
│   │   ├── variables.html  # Módulo 4: Detección y selección de variables
│   │   ├── discrete.html   # Módulos 5, 6, 7: Variable Discreta y prueba Chi-cuadrado
│   │   ├── continuous.html # Módulos 8, 9, 10: Variable Continua y pruebas de Normalidad
│   │   ├── distributions.html # Módulo 11: Selección final y justificación académica
│   │   ├── simulation.html # Módulos 12, 13: Generación y Comparación Real vs Simulado
│   │   ├── matchmaking.html# Módulos 14, 15, 16, 17: Simulador de Matchmaking en vivo
│   │   └── report.html     # Módulo 18: Informe académico completo de 20 secciones
│   └── static/
│       ├── css/style.css   # Estilos CS:GO dark-mode
│       └── js/main.js      # Configuraciones globales de Plotly y utilidades
├── data/
│   ├── dataset.csv         # Dataset representativo activo (50,000 registros, 33 columnas)
│   ├── mm_master_demos.csv # Demos oficiales de Matchmaking Valve (Ranks 1-18, Daño, Tiempos)
│   └── esea_meta_demos...  # Metadatos de rondas ESEA (Ronda, Duración de ronda)
├── requirements.txt        # Dependencias de Python
├── README.md               # Documentación integral
└── run.py                  # Script de ejecución
```

---

## 🚀 Guía de Instalación y Ejecución

### 1. Clonar o abrir el directorio del proyecto
Asegúrate de estar en la carpeta raíz del proyecto:
```bash
cd "c:\Users\willy\Downloads\Modelacion-Matchmaking csgo"
```

### 2. Crear y activar el entorno virtual
En Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
En Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar las dependencias
```bash
pip install -r requirements.txt
```

### 4. Detección y Descarga del Dataset
El sistema detecta automáticamente los archivos CSV ubicados en `data/` o en el caché de KaggleHub. Si deseas descargar la última versión del dataset oficial directamente desde Kaggle:
```bash
python -c "import kagglehub; path = kagglehub.dataset_download('skihikingkevin/csgo-matchmaking-damage'); print('Descargado en:', path)"
```
O bien, utiliza el botón **"Sincronizar KaggleHub"** directamente en la interfaz del explorador de dataset.

### 5. Iniciar la aplicación Flask
```bash
python app/app.py
```
O alternativamente:
```bash
python run.py
```

La aplicación se iniciará en modo interactivo en:
👉 **`http://127.0.0.1:5000`**

---

## 📊 Descripción de los 18 Módulos Implementados

1. **Contexto del Matchmaking (Dashboard):** Visualización interactiva del flujo del jugador, distinción epistemológica entre datos reales, parámetros estimados, modelos teóricos y reglas de emparejamiento.
2. **Explorador del Dataset:** Selector de archivos CSV (soporta `dataset.csv`, `mm_master_demos.csv`, `esea_meta_demos`, etc.), visualización de primeras/últimas filas, dimensiones, conteo de nulos, duplicados y estadísticas básicas.
3. **Limpieza de Datos:** Operaciones controladas sobre copias aisladas (eliminación de duplicados, tratamiento de valores nulos mediante eliminación o imputación, filtrado IQR de outliers y límites de rango). Tabla comparativa dinámica ANTES vs DESPUÉS.
4. **Selección de Variables:** Análisis de cardinalidad y significado conceptual. Detección de variables discretas (`att_rank`, `round`, `ct_alive`) y continuas (`seconds`, `ct_eq_val`, `t_eq_val`, `avg_match_rank`).
5. **Variable Discreta:** Métricas descriptivas, tabla de frecuencias completa, gráficos de barras e histogramas discretos de probabilidades.
6. **Ajuste de Distribuciones Discretas:** Estimación de parámetros para Poisson, Binomial, Binomial Negativa y Geométrica. Cálculo de frecuencias esperadas con agrupación de colas para validez asintótica.
7. **Hipótesis Estadísticas:** Contraste de bondad de ajuste mediante Chi-cuadrado ($\chi^2$). Nivel de significancia configurable ($\alpha = 0.01, 0.05, 0.10$). Interpretación formal rigurosa ("no se rechaza $H_0$").
8. **Variable Continua:** Momentos muestrales, percentiles (P5 a P95), histograma continuo, estimación de densidad por kernel (KDE), Boxplot y función acumulada empírica (ECDF).
9. **Distribuciones Continuas:** Estimación por máxima verosimilitud (MLE) para Normal, Exponencial, Lognormal, Gamma, Weibull y Uniforme. Prueba de Kolmogorov-Smirnov (KS) y superposición de curvas de densidad sobre el histograma.
10. **Pruebas de Normalidad:** Shapiro-Wilk (con submuestreo metodológico si $N > 5000$), Anderson-Darling con tabla de valores críticos al 15%, 10%, 5%, 2.5% y 1%, prueba $K^2$ de D'Agostino-Pearson y gráfico Cuantil-Cuantil (Q-Q Plot). Incluye advertencia explícita sobre la potencia de prueba en muestras masivas.
11. **Selección Multiobjetiva:** Puntaje ponderado (Ajuste 0.45 + Momentos 0.30 + Soporte/Contexto 0.25) restringido a familias viables, con ranking completo, contraste de momentos y justificación contextual para el matchmaking.
12. **Generación de Variables Aleatorias:** Generador pseudoaleatorio para 10,000 observaciones con semilla configurable. Documentación detallada de los algoritmos (Knuth, Marsaglia-Tsang, Box-Muller, Transformada Inversa); BTPE de NumPy se usa solo como respaldo cuando λ ≥ 30 o n·p ≥ 30, por coste O(λ) y O(n) por draw.
13. **Comparación Real vs Simulada:** Tablas comparativas lado a lado con errores relativos porcentuales, histogramas y densidades superpuestos, y pruebas estadísticas de dos muestras (KS de dos muestras propio, porque SciPy 1.18 rechaza `stats.kstest(data, 'norm', args=...)`, y Mann-Whitney U).
14. **Simulador de Matchmaking:** Interfaz visual con animación de jugadores ingresando a la cola (`Jugador 001 → Cola...`) y banner dinámico `¡MATCH ENCONTRADO! PARTIDA #001`.
15. **Simulación de Cola:** Proceso estocástico de llegadas, generación de perfiles de jugador muestreados de las distribuciones ajustadas y encolamiento con control de saturación.
16. **Algoritmo de Compatibilidad:** Regla matemática basada en la distancia euclidiana normalizada $d(p_1, p_2) \le \theta$ con umbral de tolerancia configurable.
17. **Resultados de la Simulación:** Métricas de partidas creadas, jugadores en espera, tasa de emparejamiento, tiempos promedio/máximos de espera, gráficos temporales del tamaño de cola e histograma de esperas.
18. **Marco Teórico y Ecuaciones Matemáticas:** Módulo interactivo con la formulación axiomática completa de Kolmogórov, deducciones por máxima verosimilitud (MLE), ecuaciones de PMF/PDF/CDF, demostración del teorema de la transformada inversa, algoritmo Box-Muller, modelación del matchmaking por distancias normalizadas y glosario de notación en KaTeX.
19. **Informe Automático (20 Secciones):** Generación en tiempo real de un reporte técnico universitario completo con los números, parámetros y decisiones reales obtenidas de la muestra. Exportable a Markdown (.md) y preparado para impresión o guardado en PDF.

---

## ⚠️ Consideraciones Estadísticas Importantes

- **Potencia en Muestras Grandes:** Al trabajar con muestras empíricas de 50,000+ registros, los tests formales de bondad de ajuste ($p < 0.05$) rechazan la hipótesis nula ante divergencias numéricas infinitesimales. Por ello, la validación se complementa indispensablemente con el análisis visual (superposición de densidad ajustada, Q-Q plot y convergencia de momentos).
- **Sobredispersión frente a subdispersión (consecuencia en los datos reales):** si la varianza supera a la media, $\text{Var}(X) > E(X)$, la *Poisson* subestima la variabilidad de las colas y la *Binomial Negativa* ofrece la parametrización correcta. Pero laBinomial Negativa exige $\text{Var}(X) \ge E(X)$, de modo que si la muestra es **subdispersa** esa familia queda inadmisible por estructura, antes de mirar cualquier estadístico. En este dataset `att_rank` presenta $\text{Var}/\text{E} = 0.9626 < 1$, es decir subdispersa, y por eso se descarta. El ajuste de Poisson usa el lámite $\lambda \to 0.999$ del límite Poisson de la Binomial Negativa como solución coherente.
- **Aproximación Académica:** Las reglas de compatibilidad y formación de partidas representan una abstracción pedagógica orientada a ilustrar conceptos estocásticos y de colas, no la implementación comercial propietaria de Valve.
