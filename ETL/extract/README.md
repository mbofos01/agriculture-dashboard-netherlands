# Extract Microservice

## Brief Explanation

This container runs a cronjob every month to check if newer data is available in the web. If there are, it fetches them, stores them the `data` directory and awakens the `Transform` container. The `communication` library is used to communicate through the `RabbitMQ` message broker.

## Built-Upon

```
python:3.9-slim
```

## Dependencies
```
faostat==1.1.2
psycopg2-binary==2.9.10
SQLAlchemy==2.0.36
PuLP==2.9.0
pandas==2.2.3
pika==1.3.2
cbsodata==1.3.5
```
