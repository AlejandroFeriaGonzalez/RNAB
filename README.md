# Proyecto de Redes Neuronales y Algoritmos Bioinspirados

Este repositorio contiene el desarrollo de tres módulos principales aplicados al sector transporte, como parte del curso de Redes Neuronales y Algoritmos Bioinspirados (RNAB) de la Universidad Nacional de Colombia.

## Estructura del Proyecto

- `notebooks/`: Notebooks con experimentos, análisis y desarrollo de modelos.
  - `trabajo1_Passengers.ipynb`: Predicción de demanda de pasajeros (series de tiempo).
  - `modelo 2 distracciones.ipynb`: Clasificación de distracciones al conducir (visión por computador).
  - `modelo3 recomendaciones.ipynb`: Recomendaciones personalizadas de destinos turísticos.
- `api/`: API en FastAPI para exponer los modelos como servicios web.
  - `src/services/`: Implementación de los servicios de predicción y recomendación.
  - `src/static/`: Archivos estáticos y plantillas HTML para la interfaz web.
- `blog/`: Recursos y datos para la documentación y visualización.
- `requirements.txt`: Dependencias principales del proyecto.

## Módulos Principales

### 1. Predicción de Demanda de Transporte

- Modelo LSTM para predecir la demanda diaria de pasajeros en el sistema de transporte público de Madrid.
- Pipeline de preprocesamiento, escalamiento, entrenamiento y evaluación.
- Visualización de resultados y métricas de desempeño.

### 2. Clasificación de Distracciones al Conducir

- Modelos CNN y transfer learning (MobileNetV2, EfficientNet, CNN desde cero) para clasificar imágenes de conductores según el tipo de distracción.
- Análisis de resultados, matriz de confusión y visualización de ejemplos.

### 3. Recomendador de Destinos Turísticos

- Sistema híbrido de recomendación (colaborativo + basado en contenido) para sugerir destinos personalizados.
- Simulación y validación cruzada de recomendaciones.
- Interfaz interactiva para recibir feedback y mejorar el sistema.

## API

La carpeta `api/` contiene una API en FastAPI que expone los modelos entrenados como servicios web para integración y pruebas.

### Ejecución de la API

1. Instala las dependencias:
   ```bash
   pip install -r api/requirements.txt
   ```
2. Ejecuta la API:
   ```bash
   python api/src/main.py
   ```
3. Accede a la interfaz web en [http://localhost:8000](http://localhost:8000).

## Requisitos

- Python 3.12
- Paquetes: numpy, pandas, scikit-learn, torch, tensorflow, fastapi, etc.

## Créditos

Desarrollado por Alejandro Feria para la asignatura de Redes Neuronales y Algoritmos Bioinspirados, Universidad Nacional de Colombia.

