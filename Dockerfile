FROM python:3.10-slim

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app

ENV MLFLOW_TRACKING_URI="https://52.73.27.252"

ENV MLFLOW_TRACKING_USERNAME="team40"
ENV MLFLOW_TRACKING_PASSWORD="472172Mna"

ENV MLFLOW_TRACKING_INSECURE_TLS=true
ENV CURL_CA_BUNDLE=""

ENV MODEL_VERSION=9

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
