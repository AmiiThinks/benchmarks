# syntax=docker/dockerfile:1
FROM python:3.8
WORKDIR /app
COPY config/ config/
COPY logs/ logs/
COPY models/ models/
COPY src/ src/
COPY sygus_string_tasks/ sygus_string_tasks/
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /app/src
CMD python3 bustle_batch_encoded_main.py
