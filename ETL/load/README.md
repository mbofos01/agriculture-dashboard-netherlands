# Load Microservice

## Brief Explanation

When this container is awaken it loads the new data transformed by the `Transform` container. After that the `Server` container is awaken .The `communication` library is used to communicate through the `RabbitMQ` message broker.

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
```
