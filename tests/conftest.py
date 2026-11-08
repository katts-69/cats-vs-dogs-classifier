"""Shared test setup. pytest runs this automatically before the tests."""
from pathlib import Path

import pytest

from src.predict import Predictor, find_model

IMAGES = Path(__file__).parent / "images"


@pytest.fixture(scope="session")
def predictor():
    """Load the model once for all tests (local file, or downloaded from Hugging Face in CI)."""
    return Predictor(find_model(), device="cpu")
