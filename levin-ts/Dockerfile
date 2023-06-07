FROM python:3.10.2
ENV PIP_ROOT_USER_ACTION=ignore
COPY bilevin /bilevin
RUN pip install --upgrade pip
RUN pip install torch==1.13.1
RUN cd /bilevin && pip install -r requirements.txt
