FROM python:3.6-slim-stretch

ADD requirements.txt requirements.txt

RUN apt-get update

RUN python -m pip install -r requirements.txt 

WORKDIR /workspace/src
COPY . .
EXPOSE 8888
