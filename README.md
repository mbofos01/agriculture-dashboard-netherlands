# Agriculture and Weather Dashboard

## How to run

```bash
  $ docker-compose up --build
  visit: localhost:8050
```

## Container Structure

- ETL process has been broken down to three different containers
- These containers communicate throught the RabbitMQ broker

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

    style R fill:#f9f,stroke:#333,stroke-width:2px,color:#000;
    style E fill:#bbf,stroke:#333,stroke-width:2px,color:#000;
    style T fill:#bbf,stroke:#333,stroke-width:2px,color:#000;
    style L fill:#bbf,stroke:#333,stroke-width:2px,color:#000;
    style D fill:#bfb,stroke:#333,stroke-width:2px,color:#000;
    style W fill:#FF6961,stroke:#333,stroke-width:2px,color:#000;
    style S fill:#ADD8E6,stroke:#333,stroke-width:2px,color:#000;
```
