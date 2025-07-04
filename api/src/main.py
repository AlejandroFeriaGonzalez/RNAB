import pathlib

from fastapi import FastAPI, File, Request, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.services.driving_classification import predict_image, PredictionResult
from src.services.travel_recommendations import (
    get_travel_recommendation_api,
    submit_feedback_api,
    get_available_options,
    TravelRecommendationRequest,
    FeedbackRequest,
)
from src.services.demand_prediction import predict_demand_from_upload, DemandPredictionResponse

SRC = pathlib.Path(__file__).parent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=SRC / "static"), name="static")
templates = Jinja2Templates(directory=SRC / "static/")


@app.get("/")
async def index():
    return RedirectResponse(url="/demand-prediction")


@app.get("/demand-prediction", response_class=HTMLResponse)
async def demand_prediction(request: Request):
    return templates.TemplateResponse(request, "demand-prediction.html")


@app.get("/driving-classification", response_class=HTMLResponse)
async def driving_classification(request: Request):
    return templates.TemplateResponse(request, "driving-classification.html")


@app.get("/travel-recommendations", response_class=HTMLResponse)
async def travel_recommendations(request: Request):
    return templates.TemplateResponse(request, "travel-recommendations.html")


@app.post("/api/driving-classification", response_model=PredictionResult)
async def api_driving_classification(file: UploadFile = File(...)):
    """
    Endpoint to classify a driver's image.
    Accepts an image file and returns the predicted class and probability.
    """
    image_bytes = await file.read()
    prediction = predict_image(image_bytes)
    return JSONResponse(content=prediction)


@app.post("/api/travel-recommendations")
async def api_travel_recommendations(request: TravelRecommendationRequest):
    """
    Endpoint para obtener recomendaciones de viaje personalizadas.
    """
    try:
        recommendations = get_travel_recommendation_api(request)
        return JSONResponse(content=recommendations.dict())
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error al generar recomendaciones: {str(e)}"},
        )


@app.post("/api/travel-recommendations/feedback")
async def api_travel_feedback(feedback: FeedbackRequest):
    """
    Endpoint para enviar feedback sobre las recomendaciones.
    """
    try:
        result = submit_feedback_api(feedback)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Error al procesar feedback: {str(e)}"}
        )


@app.get("/api/travel-recommendations/options")
async def api_travel_options():
    """
    Endpoint para obtener las opciones disponibles para el formulario.
    """
    try:
        options = get_available_options()
        return JSONResponse(content=options)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Error al obtener opciones: {str(e)}"}
        )

@app.post("/api/demand-prediction", response_model=DemandPredictionResponse)
async def api_demand_prediction(
    file: UploadFile = File(...),
    days_to_predict: int = Form(30),
    sequence_length: int = Form(45)
):
    """
    Endpoint para predecir la demanda de transporte basado en datos históricos.
    
    Args:
        file: Archivo CSV con datos históricos (columnas: 'Month', '#Passengers')
        days_to_predict: Número de días a predecir (1-365)
        sequence_length: Longitud de secuencia para el modelo LSTM
    
    Returns:
        DemandPredictionResponse: Respuesta con predicciones y metadatos
    """
    # Validar tipo de archivo
    if not file.filename.endswith('.csv'):
        return JSONResponse(
            status_code=400,
            content={"error": "El archivo debe ser de tipo CSV"}
        )
    
    # Validar parámetros
    if not (1 <= days_to_predict <= 365):
        return JSONResponse(
            status_code=400,
            content={"error": "Los días a predecir deben estar entre 1 y 365"}
        )
    
    if not (1 <= sequence_length <= 100):
        return JSONResponse(
            status_code=400,
            content={"error": "La longitud de secuencia debe estar entre 1 y 100"}
        )
    
    try:
        # Leer el contenido del archivo
        file_content = await file.read()
        
        # Llamar a la función de predicción
        result = await predict_demand_from_upload(
            file_content=file_content,
            days_to_predict=days_to_predict,
            sequence_length=sequence_length
        )
        
        if result.success:
            return result
        else:
            return JSONResponse(
                status_code=400,
                content={"error": result.message}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error interno del servidor: {str(e)}"}
        )