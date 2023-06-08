FROM python:3.10.2
WORKDIR /src
COPY src /src
RUN pip install --upgrade pip
RUN pip install torch==1.13.1
RUN pip install -r requirements.txt

CMD /bin/bash -c "time /src/augmented/augmented_bustle_experiments.sh"
