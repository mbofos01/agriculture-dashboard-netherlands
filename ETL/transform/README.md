# Transform Microservice

## Brief Explanation

When this container is awaken it transforms the new data fetched by the `Extract` container. After that the `Load` container is awaken .The `communication` library is used to communicate through the `RabbitMQ` message broker.

## Built-Upon

```
python:3.9-slim
```

## Dependencies
```
pandas==2.2.3
pika==1.3.2
```
