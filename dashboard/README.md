# Server Microservice

## Brief Explanation

This container hosts the main Dashboard App. When all tables are present, our <a href="https://dash.plotly.com/">Dash App</a> starts and port `8050` is exposed. Setting on/off the debug mode of the server can be changed from the `docker-compose.yml` file. 

You can <a href="http://localhost:8050">visit the dashboard here.</a>

## Built-Upon

```
python:3.11
```

## Dependencies
```
geopy==2.4.1
openmeteo_requests==1.3.0
requests-cache==1.2.1
retry-requests==2.0.0
scipy==1.14.1
pandas==2.2.3
plotly==5.24.1
dash==2.18.1
scikit-learn==1.5.2
dash-bootstrap-components==1.6.0
numpy==2.1.2
dash-daq==0.5.0
ipython==8.28.0
SQLAlchemy==2.0.36
psycopg2==2.9.10
pika==1.3.2
geopandas==1.0.1
folium==0.17.0
```
