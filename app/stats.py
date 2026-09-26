"""
Módulo de compatibilidad para estadísticas.
Reexporta todas las funciones desde app/statistics.py.
"""

from statistics import (
    chi_square_test,
    ks_test,
    shapiro_test,
    anderson_test,
    dagostino_test,
    ks_2samp_test,
    mann_whitney_test,
    qq_plot_data,
    interpret_pvalue
)

__all__ = [
    'chi_square_test',
    'ks_test',
    'shapiro_test',
    'anderson_test',
    'dagostino_test',
    'ks_2samp_test',
    'mann_whitney_test',
    'qq_plot_data',
    'interpret_pvalue'
]
