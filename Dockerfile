# GreenWindow forecast API on CPU.
#   docker build -t greenwindow-api .
#   docker run -p 8000:8000 greenwindow-api      # then GET /forecast
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GREENWINDOW_ASSETS=/app/assets

WORKDIR /app

# CPU-only torch keeps the image near 1 GB instead of ~5 GB with CUDA
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[serve]"

COPY assets ./assets

RUN useradd --create-home app
USER app

EXPOSE 8000
CMD ["uvicorn", "greenwindow.api:app", "--host", "0.0.0.0", "--port", "8000"]
