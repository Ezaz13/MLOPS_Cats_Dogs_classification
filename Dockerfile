FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir --default-timeout=100 --retries 10 -r requirements-prod.txt

COPY . .
COPY models/model_export /app/model

EXPOSE 5000

CMD ["python", "src/model_serving/app.py"]
