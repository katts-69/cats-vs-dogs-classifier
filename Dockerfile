# A container image for the REST API: a small Linux system with Python, the libraries, the code
# and the trained model, all packed together so it runs the same on any computer with Docker.
#
# Build:  docker build -t cats-vs-dogs-api .
# Run:    docker run --rm -p 8000:8000 cats-vs-dogs-api
# Then open http://localhost:8000/docs

# Start from an official, minimal Linux image that already has Python 3.11
FROM python:3.11-slim

# Print Python output straight away, and don't write .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install libraries first: Docker caches this step, so changing the code later doesn't
# reinstall PyTorch on every build
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Copy only the code the API needs (see .dockerignore for what is left out)
COPY src/ src/

# Download the trained model from Hugging Face into the image, so the container works offline
RUN python -c "from src.predict import find_model; print('model at', find_model())"

# Don't run as the all-powerful root user: if the server were ever attacked, the attacker gets less
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Use only the model saved in the image; don't contact Hugging Face when the container starts
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

# Docker checks this every 30 s and marks the container unhealthy if the API stops answering
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
