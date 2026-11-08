"""Tests for predictions: the outputs make sense, real photos are classified correctly, and
unusual images (grayscale, transparent, tiny, odd shapes) don't crash the app.

Run from the project folder:
    python -m pytest
"""
import math

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import IMAGE_SIZE, eval_transform
from src.model import CLASS_NAMES, build_model
from tests.conftest import IMAGES


def random_image(width=300, height=200, mode="RGB", seed=0):
    """A noise image, so tests don't depend on any particular photo."""
    rng = np.random.default_rng(seed)
    channels = {"RGB": 3, "RGBA": 4}.get(mode)
    shape = (height, width, channels) if channels else (height, width)
    return Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8), mode)


# --- The output is a valid probability for each class -------------------------------------

def test_returns_one_probability_per_class(predictor):
    result = predictor.predict(random_image())
    assert list(result) == CLASS_NAMES


def test_probabilities_add_up_to_one(predictor):
    result = predictor.predict(random_image())
    assert math.isclose(sum(result.values()), 1.0, abs_tol=1e-5)
    assert all(0.0 <= p <= 1.0 for p in result.values())


def test_same_image_gives_same_answer(predictor):
    image = random_image()
    assert predictor.predict(image) == pytest.approx(predictor.predict(image))


# --- Real photos are classified correctly -------------------------------------------------

@pytest.mark.parametrize("filename, expected", [("cat.jpg", "Cat"), ("dog.jpg", "Dog")])
def test_real_photo_is_classified_correctly(predictor, filename, expected):
    result = predictor.predict(IMAGES / filename)  # a file path works as well as a PIL image
    assert max(result, key=result.get) == expected
    assert result[expected] > 0.9, f"right answer but unsure: {result}"


def test_flipped_photo_gets_same_answer(predictor):
    """Training used mirrored photos, so a mirrored cat should still be a cat."""
    flipped = Image.open(IMAGES / "cat.jpg").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    result = predictor.predict(flipped)
    assert result["Cat"] > 0.9


# --- Unusual images don't crash the app ---------------------------------------------------

@pytest.mark.parametrize(
    "image",
    [
        Image.open(IMAGES / "cat.jpg").convert("L"),  # grayscale (black and white)
        random_image(mode="RGBA"),  # transparent PNG
        random_image(mode="L"),  # grayscale noise
        random_image(10, 10),  # tiny
        random_image(2000, 50),  # very wide panorama
        random_image(50, 2000),  # very tall
    ],
    ids=["grayscale-photo", "rgba", "grayscale-noise", "tiny", "very-wide", "very-tall"],
)
def test_unusual_images_do_not_crash(predictor, image):
    result = predictor.predict(image)
    assert math.isclose(sum(result.values()), 1.0, abs_tol=1e-5)


def test_grayscale_cat_is_still_a_cat(predictor):
    result = predictor.predict(Image.open(IMAGES / "cat.jpg").convert("L"))
    assert max(result, key=result.get) == "Cat"


# --- Building blocks ----------------------------------------------------------------------

def test_eval_transform_makes_model_sized_tensor():
    tensor = eval_transform(random_image(640, 480))
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32


def test_model_outputs_one_score_per_class():
    model = build_model(pretrained=False).eval()  # random weights: tests the shape, not accuracy
    with torch.no_grad():
        output = model(torch.zeros(4, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert output.shape == (4, len(CLASS_NAMES))
