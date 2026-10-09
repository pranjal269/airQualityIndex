FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY data/processed/ ./data/processed/
COPY models/ ./models/
COPY docs/ ./docs/
COPY *.md ./

ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/7_generate_predictions.py"]