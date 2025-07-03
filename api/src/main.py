import pathlib

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.services.driving_classification import predict_image, PredictionResult
from src.services.travel_recommendations import recomendar_por_usuario_y_preguntas

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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


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
async def api_travel_recommendations():
    pass