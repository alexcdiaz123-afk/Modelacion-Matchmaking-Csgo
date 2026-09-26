"""
Script para descargar el dataset de CS:GO desde Kaggle.
"""

import kagglehub
import os
import shutil

def download_dataset():
    """Descarga el dataset y lo mueve a la carpeta data/."""
    print("Descargando dataset de CS:GO desde Kaggle...")
    
    # Descargar dataset
    path = kagglehub.dataset_download("skihikingkevin/csgo-matchmaking-damage")
    print(f"Dataset descargado en: {path}")
    
    # Buscar archivos CSV
    csv_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print("No se encontraron archivos CSV en el dataset.")
        return None
    
    # Crear carpeta data si no existe
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Copiar archivos CSV a la carpeta data
    for csv_file in csv_files:
        dest = os.path.join(data_dir, os.path.basename(csv_file))
        shutil.copy2(csv_file, dest)
        print(f"Archivo copiado: {dest}")
    
    print(f"\nDataset listo en: {data_dir}")
    return data_dir

if __name__ == '__main__':
    download_dataset()
