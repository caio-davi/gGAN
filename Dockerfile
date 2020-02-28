FROM python:3.6-slim-stretch

ADD requirements.txt requirements.txt

RUN apt-get update

RUN apt-get install -y graphviz

RUN pip install --upgrade pip

RUN python -m pip install -r requirements.txt 

WORKDIR /gGAN
COPY . .
EXPOSE 8888
