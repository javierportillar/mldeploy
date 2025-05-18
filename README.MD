# Proyecto de Predicción de Órdenes

Este proyecto es una aplicación web en Flask que carga un modelo de regresión entrenado (`model_admit5.pkl`), aplica el preprocesado de escalado y muestra predicciones de **Total de órdenes a entregar** según cinco características.
Este modelo se encuentra enn el siguiente link: https://colab.research.google.com/drive/1Ekuuou8zDQCKtra_lWdA12kSD7nAV8U9
## Estructura del Proyecto
proyecto/
├── app.py                  # Código principal de la aplicación Flask
├── model_admit5.pkl        # Modelo serializado (pickle)
├── requirements.txt        # Dependencias del proyecto
├── data/
│   └── predictions.csv     # Archivo donde se guardan las predicciones
├── static/
│   └── css/style.css       # Estilos CSS
└── templates/
└── index.html          # Plantilla HTML principal

## Requisitos Previos

- Python 3.8 o superior
- Git (opcional, para clonar el repositorio)

## Instalación y Ejecución

1. **Clonar el repositorio** (o copiar los archivos a tu máquina):
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd proyecto

2. **Crear y activar un entorno virtual:**
    En macOS/Linux:
    python3 -m venv env
    source env/bin/activate

    En Windows (PowerShell):
    python -m venv env
    .\\env\\Scripts\\Activate.ps1

3. **Actualizar pip (opcional, recomendado):**
    pip install --upgrade pip

4. **Instalar dependencias:**
    pip install -r requirements.txt (Instala automáticamente)
    pip install flask pandas numpy scikit-learn (Instala manualmente)

5. **Ejecutar la aplicación:**
    python app.py

* Serving Flask app 'app'
* Debug mode: on
* Running on http://127.0.0.1:5000