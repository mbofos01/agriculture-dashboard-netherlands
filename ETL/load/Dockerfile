FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 

WORKDIR /app
COPY . /app

RUN pip install -r /app/requirements.txt

ENTRYPOINT [ "python", "/app/main.py" ]