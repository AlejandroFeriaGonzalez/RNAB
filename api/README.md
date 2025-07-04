# API de Predicción y Recomendaciones de Transporte

Este proyecto implementa una API en Python para realizar predicciones y recomendaciones relacionadas con el transporte, utilizando modelos de machine learning. Incluye servicios para predicción de demanda, clasificación de conducción y recomendaciones de viaje.

## Estructura del Proyecto

- `src/`: Código fuente de la API y servicios.
  - `main.py`: Punto de entrada de la API.
  - `services/`: Servicios de predicción y recomendación.
    - `demand_prediction.py`: Predicción de demanda de transporte.
    - `driving_classification.py`: Clasificación de tipo de conducción.
    - `travel_recommendations.py`: Recomendaciones de viaje personalizadas.
- `static/`: Archivos estáticos y plantillas HTML para la interfaz web.
- `requirements.txt`: Dependencias del proyecto.

## Instalación

1. Clona este repositorio y navega a la carpeta `api`.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Ejecuta la API con:
```bash
python src/main.py
```
La API estará disponible en el puerto configurado (por defecto 8000).

## Servicios Disponibles

- **Predicción de demanda**: Predice la demanda de transporte en función de datos históricos.
- **Clasificación de conducción**: Clasifica imágenes según el tipo de conducción (segura, distracción, etc.).
- **Recomendaciones de viaje**: Ofrece recomendaciones personalizadas de destinos y rutas.

## Docker

Puedes construir y ejecutar la API usando Docker:
```bash
docker build -t transporte-api .
docker run -p 8000:8000 transporte-api
```

## Créditos

Desarrollado por Alejandro Feria para la asignatura de Redes Neuronales y Algoritmos Bioinspirados, Universidad Nacional de Colombia.
