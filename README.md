# Agriculture and Weather Dashboard

## How to build & run

```bash
  $ docker-compose up --build
```

- Once the build is complete, visit your application in the browser: http://localhost:8050 or http://127.0.0.1:8050


- This will launch your Docker services and open the application on port 8050

- You can visit http://localhost:15672 (username: guest, password:guest) to view the RabbitMQ management dashboard

- All requirements are handled by the respective Dockerfiles

## Project Structure

- ETL process has been broken down to three different microservices that communicate through RabbitMQ.
- Extract fetches data from the web or legacy files and enables Transform
- Transform is supposed to transform raw data and enable Load
- Load inputs the new data to the database

```mermaid
      graph TD
        legacy -->|depends_on: service_healthy| database
      
        extract -->|depends_on: service_healthy| rabbitmq
        extract -->|depends_on: service_started| transform
        transform -->|depends_on: service_healthy| rabbitmq
        transform -->|depends_on: service_started| load
        load -->|depends_on: service_healthy| rabbitmq
        server -->|depends_on: service_healthy| rabbitmq
        server -->|depends_on: service_started| database
        server -->|depends_on: service_started| load
        server -->|depends_on: service_started| transform
        server -->|depends_on: service_started| extract


      style rabbitmq fill:#FF0000,stroke:#333,stroke-width:2px,color:#000;
      style extract fill:#50C878,stroke:#333,stroke-width:2px,color:#000;
      style transform fill:#e1c751,stroke:#333,stroke-width:2px,color:#000;
      style load fill:#6588d0,stroke:#333,stroke-width:2px,color:#000;
      style database fill:#008bb9,stroke:#333,stroke-width:2px,color:#000;
      style server fill:#f4ce8d,stroke:#333,stroke-width:2px,color:#000;
      style legacy fill:#8f5ec4,stroke:#333,stroke-width:2px,color:#000;
```
