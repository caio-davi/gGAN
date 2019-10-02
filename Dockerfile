FROM python:3.6-slim-stretch

ADD requirements.txt requirements.txt

RUN python -m pip install -r requirements.txt 

WORKDIR /workspace
COPY . .
EXPOSE 8888
