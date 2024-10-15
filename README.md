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

    graph TD;
		
		E -->|communicates with| W[Web]

    E[Extract] -->|depends on & uses| R[RabbitMQ]
    T[Transform] -->|depends on & uses| R
    L[Load] -->|depends on & uses| R
    
    E --> |depends on| T
    T --> |depends on| L
    
    E -->|shares| S[Shared Space]
    T -->|shares| S[Shared Space]


    subgraph Database
        D[Postgres Database]
    end

  
    L -->|connects to| D
    L -->|shares| S[Shared Space]
    
     SE[Dashboard App] -->|connects to| D

    style R fill:#f9f,stroke:#333,stroke-width:2px,color:#000;
    style E fill:#bbf,stroke:#333,stroke-width:2px,color:#000;
    style T fill:#bbf,stroke:#333,stroke-width:2px,color:#000;
    style L fill:#bbf,stroke:#333,stroke-width:2px,color:#000;
    style D fill:#bfb,stroke:#333,stroke-width:2px,color:#000;
    style W fill:#FF6961,stroke:#333,stroke-width:2px,color:#000;
    style S fill:#ADD8E6,stroke:#333,stroke-width:2px,color:#000;
    style SE fill:#FFA500,stroke:#333,stroke-width:2px,color:#000;
```
