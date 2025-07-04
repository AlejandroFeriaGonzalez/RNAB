import pathlib

from fastapi import FastAPI, File, Request, UploadFile
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
