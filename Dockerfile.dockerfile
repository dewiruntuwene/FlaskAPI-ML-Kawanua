# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install flask
COPY app.py app.py
EXPOSE 5000


CMD [ "python3", "/python-docker/app.py"]