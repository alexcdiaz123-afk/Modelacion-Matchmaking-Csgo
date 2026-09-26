"""
Script de inicio rápido para la aplicación.
"""

import os
import sys

# Agregar la carpeta app al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app import app

if __name__ == '__main__':
    print("=" * 60)
    print("Simulador Estadístico de Matchmaking — CS:GO")
    print("=" * 60)
    print("\nIniciando servidor Flask...")
    print("La aplicación estará disponible en: http://127.0.0.1:5000")
    print("\nPresione Ctrl+C para detener el servidor.\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
