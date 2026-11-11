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

# Photos from strangers are untrusted input. Only these common photo formats are decoded:
# Pillow can also read rarer formats (TIFF, PDF, EPS, FlashPix...), and that is where most of
# its security bugs have been found, so refusing them shrinks what an attacker can reach.
ALLOWED_FORMATS = ["JPEG", "PNG", "WEBP", "GIF", "BMP"]

# A small file can claim enormous dimensions and use gigabytes of memory once decoded
# (a "decompression bomb"). 64 million pixels still allows photos from 50-megapixel phones.
MAX_PIXELS = 64_000_000

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
        # open() only reads the header, so the size can be checked before any pixels are decoded
        image = Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS)
        if image.width * image.height > MAX_PIXELS:
            raise HTTPException(status_code=413, detail="Image has too many pixels (limit 64 megapixels).")
        image.load()  # actually decode it, so broken files fail here and not later
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=400,
            detail="That file is not an image I can read. Use JPEG, PNG, WEBP, GIF or BMP.",
        )

    probabilities = models["predictor"].predict(image)
    label = max(probabilities, key=probabilities.get)
    return Prediction(label=label, confidence=probabilities[label], probabilities=probabilities)
