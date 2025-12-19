# Agriculture Dashboard

## How to build

```bash

$ docker-compose up --build

```

After building is over you can visit http://localhost:8050 to view our dashboard.

## Structure

Our Dashboard application consists of multiple microservices that communicate with each other. The architecture of our system is shown in the following figure. Each block represents a microservice (a Docker container that serves a purpose on our system). We can easily observe that the ETL process is broken down into three distinct microservices.

```mermaid
%%{init: {"theme": "default", "themeVariables": {"background":"#ffffff"}}}%%
    graph BT
    subgraph Network_web[<b>Network: web</b>]
        rabbitmq["<b>rabbitmq</b>"]
        database["<b>database</b>"]
        extract["<b>extract</b>"]
        transform["<b>transform</b>"]
        load["<b>load</b>"]
        dashboard["<b>dashboard</b>"]
        legacy["<b>legacy</b>"]
    end
    style Network_web fill:#e4c981ff
    rabbitmq["<b>rabbitmq</b>"]
    database["<b>database</b>"]
    extract["<b>extract</b>"]
    transform["<b>transform</b>"]
    load["<b>load</b>"]
    dashboard["<b>dashboard</b>"]
    legacy["<b>legacy</b>"]
    extract -- depends_on (service_healthy) --> rabbitmq
    linkStyle 0 stroke-width:2,stroke-dasharray:5 5
    extract -- depends_on (service_started) --> transform
    linkStyle 1 stroke-width:2,stroke-dasharray:
    transform -- depends_on (service_healthy) --> rabbitmq
    linkStyle 2 stroke-width:2,stroke-dasharray:5 5
    transform -- depends_on (service_started) --> load
    linkStyle 3 stroke-width:2,stroke-dasharray:
    load -- depends_on (service_healthy) --> rabbitmq
    linkStyle 4 stroke-width:2,stroke-dasharray:5 5
    dashboard -- depends_on (service_healthy) --> rabbitmq
    linkStyle 5 stroke-width:2,stroke-dasharray:5 5
    dashboard -- depends_on (service_started) --> database
    linkStyle 6 stroke-width:2,stroke-dasharray:
    dashboard -- depends_on (service_started) --> load
    linkStyle 7 stroke-width:2,stroke-dasharray:
    dashboard -- depends_on (service_started) --> transform
    linkStyle 8 stroke-width:2,stroke-dasharray:
    dashboard -- depends_on (service_started) --> extract
    linkStyle 9 stroke-width:2,stroke-dasharray:
    legacy -- depends_on (service_healthy) --> database
    linkStyle 10 stroke-width:2,stroke-dasharray:5 5
    style rabbitmq fill:#d89e87ff,stroke:#f16026ff,stroke-width:3,stroke-dasharray:0
    style database fill:#879ad8ff,stroke:#265cf1ff,stroke-width:3,stroke-dasharray:0
    style extract fill:#87d89fff,stroke:#4fca43ff,stroke-width:3,stroke-dasharray:0
    style transform fill:#d887d8ff,stroke:#ca43bfff,stroke-width:3,stroke-dasharray:0
    style load fill:#ad87d8ff,stroke:#6c43caff,stroke-width:3,stroke-dasharray:0
    style dashboard fill:#d88787ff,stroke:#ca4343ff,stroke-width:3,stroke-dasharray:0
    style legacy fill:#e2cfcfff,stroke:#000000ff,stroke-width:3,stroke-dasharray:0

```

## Microservices

1. Database
    
    We use the PostgreSQL relational database management system, to host our database. PostgreSQL is a free and open-source project.
    
2. RabbitMQ
    
    RabbitMQ is an open-source message broker, that enables communication between microservices in our architecture. RabbitMQ creates a communication queue between two processes (services) allowing them to send and receive messages asynchronously. In our case, the exchanged messages are usually tasks assigned from one microservice to another. By visiting the <a href="http://localhost:15672/">local RabbitMQ Management Inteface</a> we can see more details and analytics about the communication of our microservices.

3. Legacy
    
    Legacy is one microservice that is only executed once on start-up. Since we use some legacy data that require specific handling (preprocessing, file combination etc) we created this microservice that loads said legacy data to our database.
    
4. Extract
    
    The Extract microservice is the main microservice of the ETL pipeline. A cronjob is running in that container every month (as of now the frequency of the cronjob is set on 10 minutes for testing purposes) checking whether the datasets we use have been updated. If the datasets have been updated it acquires the new data and enables the Transform microservice.
    
5. Transform
    
    When enabled, the Transform microservice preprocesses the new data, and the Load microservice is enabled.
    
6. Load
    
    When enabled, the Load microservice communicates with the Database and loads the new and processed data. Then, a message is sent to the Server for the Server to load the latest data from the Database.
    
7. Server
    
    Our server is built using the Dash Python framework, which in turn is built upon the Flask micro web framework. Our server starts when all used tables are present in the Database. Then the latest data are loaded and used. The utilized data are reloaded only when new data are available - that mechanism was implemented to reduce the number of times any formatting process was running. 
    
