import pathlib
from io import BytesIO
from typing import Literal, TypedDict

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Assuming the models directory is at the root of the api project
SERVICES_PATH = pathlib.Path(__file__).parent
MODEL_PATH = SERVICES_PATH / "models" / "modelo_cnn_grayscale_balanced.h5"

# Clases del modelo
CLASS_LABELS = ['other_activities', 'safe_driving', 'talking_phone', 'texting_phone', 'turning']

# Cargar el modelo
model = load_model(MODEL_PATH)


class PredictionResult(TypedDict):
    clase_predicha: Literal['other_activities', 'safe_driving', 'talking_phone', 'texting_phone', 'turning']
    probabilidad: float


def predict_image(img_bytes: bytes, img_size=(256, 256)) -> PredictionResult:
    """
    Predice la clase de una imagen utilizando el modelo entrenado.

    Args:
        img_bytes: Bytes de la imagen a predecir.
        img_size: Tamaño de la imagen (debe coincidir con el tamaño usado en el entrenamiento).

    Returns:
        dict: Diccionario con las claves "clase_predicha" y "probabilidad".
    """
    # Cargar y preprocesar la imagen desde bytes
    img = Image.open(BytesIO(img_bytes)).convert('L')  # Convertir a escala de grises
    img = img.resize(img_size)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    # Realizar la predicción
    pred = model.predict(x)
    pred_class = np.argmax(pred)
    prob = np.max(pred)

    # Resultado
    return {
        "clase_predicha": CLASS_LABELS[pred_class],
        "probabilidad": float(prob)
    }
