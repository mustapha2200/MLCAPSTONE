FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ ./api/
COPY model/ ./model/
COPY data/raw/kaggle_train.csv ./data/raw/kaggle_train.csv
COPY data/processed/enriched_data.csv ./data/processed/enriched_data.csv

EXPOSE 5000

CMD ["gunicorn", "api.app:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]
