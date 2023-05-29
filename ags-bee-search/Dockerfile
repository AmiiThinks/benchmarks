# syntax=docker/dockerfile:1
FROM python:3.10
WORKDIR /app
COPY config/ config/
COPY logs/ logs/
COPY models/ models/
COPY results/ results/
COPY src/ src/
COPY sygus_string_tasks/ sygus_string_tasks/
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /app/src
CMD python3 bee.py -t 1 -d 0 -l 1_augment_log.log -m bustle_model_03.hdf5 -b bustle_benchmarks.txt -a 1 -p 14000000; cat ../logs/1_augment_log.log
RUN cat ../logs/1_augment_log.log
