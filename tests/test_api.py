"""Tests for the REST API in src/api.py, using FastAPI's test client (no real server needed)."""
import io

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src.api import app
from tests.conftest import IMAGES


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:  # "with" runs the startup code, which loads the model
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


@pytest.mark.parametrize("filename, expected", [("cat.jpg", "Cat"), ("dog.jpg", "Dog")])
def test_predict_photo(client, filename, expected):
    with open(IMAGES / filename, "rb") as f:
        response = client.post("/predict", files={"file": (filename, f, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == expected
    assert body["confidence"] > 0.9
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)


def test_not_an_image_is_rejected(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"hello, I am text", "text/plain")})
    assert response.status_code == 400


def test_missing_file_is_rejected(client):
    response = client.post("/predict")
    assert response.status_code == 422  # FastAPI's "the request is missing something"


def test_too_large_upload_is_rejected(client):
    huge = b"\0" * (10 * 1024 * 1024 + 1)
    response = client.post("/predict", files={"file": ("huge.jpg", huge, "image/jpeg")})
    assert response.status_code == 413


def _image_bytes(image, fmt):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_png_and_webp_photos_are_accepted(client):
    cat = Image.open(IMAGES / "cat.jpg")
    for fmt in ("PNG", "WEBP"):
        response = client.post("/predict", files={"file": (f"cat.{fmt.lower()}", _image_bytes(cat, fmt), "image/*")})
        assert response.status_code == 200, fmt
        assert response.json()["label"] == "Cat"


def test_unusual_image_formats_are_refused(client):
    # TIFF is a real image, but a format the API deliberately doesn't decode
    tiff = _image_bytes(Image.open(IMAGES / "cat.jpg"), "TIFF")
    response = client.post("/predict", files={"file": ("cat.tiff", tiff, "image/tiff")})
    assert response.status_code == 400


def test_decompression_bomb_is_refused(client):
    # 81 million black-and-white pixels: a tiny file that would need lots of memory to decode
    bomb = _image_bytes(Image.new("1", (9000, 9000)), "PNG")
    assert len(bomb) < 1024 * 1024
    response = client.post("/predict", files={"file": ("bomb.png", bomb, "image/png")})
    assert response.status_code == 413
