# ---------- Builder ----------
FROM python:3.10 AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip wheel --no-cache-dir -r requirements.txt

# ---------- Runtime ----------
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /build /wheels

RUN pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

COPY src/ src/

ENV PYTHONPATH=/app
EXPOSE 5000

CMD ["python", "src/model_serving/app.py"]
