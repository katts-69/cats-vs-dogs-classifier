"""REST API: other programs send a photo over HTTP and get the prediction back as JSON.

Run from the project folder:
    python -m uvicorn src.api:app --reload
then open http://127.0.0.1:8000/docs to try it in the browser, or call it from code:
    curl -F "file=@tests/images/cat.jpg" http://127.0.0.1:8000/predict

Endpoints:
    GET  /health   -> {"status": "ok"}                  (is the server up?)
    POST /predict  -> {"label": "Cat", "confidence": 0.99, "probabilities": {"Cat": 0.99, "Dog": 0.01}}
"""
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.model import CLASS_NAMES
from src.predict import Predictor, find_model

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB; bigger uploads are refused

models = {}


@asynccontextmanager
async def lifespan(app):
    # Load the model once when the server starts, not on every request
    models["predictor"] = Predictor(find_model(), device="cpu")
    yield
    models.clear()


app = FastAPI(
    title="Cat or Dog API",
    description="Send a photo, get back whether it shows a cat or a dog (EfficientNet-B0, 98.95% test accuracy).",
    version="1.0.0",
    lifespan=lifespan,
)


class Prediction(BaseModel):
    """The shape of a /predict answer. FastAPI uses it to check the output and document the API."""

    label: str
    confidence: float
    probabilities: dict[str, float]

    model_config = {
        "json_schema_extra": {
            "examples": [{"label": "Cat", "confidence": 0.998, "probabilities": {"Cat": 0.998, "Dog": 0.002}}]
        }
    }


@app.get("/health")
def health():
    """Lets other systems (and Docker) check that the server is running and the model is loaded."""
    return {"status": "ok", "model_loaded": "predictor" in models, "classes": CLASS_NAMES}


@app.post("/predict", response_model=Prediction)
def predict(file: UploadFile = File(..., description="A JPEG, PNG or other image file")):
    """Classify one uploaded photo as a cat or a dog."""
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (limit 10 MB).")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # actually decode it, so broken files fail here and not later
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="That file is not an image I can read.")

    probabilities = models["predictor"].predict(image)
    label = max(probabilities, key=probabilities.get)
    return Prediction(label=label, confidence=probabilities[label], probabilities=probabilities)
