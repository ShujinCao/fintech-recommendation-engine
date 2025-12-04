FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y libgomp1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir faiss-cpu

COPY . .

# Train models during build
RUN python -m src.data.generate_synthetic_data && \
    python -m src.retrieval.e5_product_encoder && \
    python -m src.features.build_features && \
    python -m src.models.train_two_tower_multimodal && \
    python -m src.models.train_ranker

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD uvicorn src.serving.app:app --host 0.0.0.0 --port $PORT

