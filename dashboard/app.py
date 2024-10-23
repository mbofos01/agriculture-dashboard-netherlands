import pika
from datetime import date, datetime, timedelta
from dash import Input, Output, State, callback_context
import dash_daq as daq  # Import dash_daq
import datetime
from IPython.display import display, HTML
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc
from dash import dcc, html
from scipy.stats import pearsonr
from scipy.stats import gmean
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go  # Import for creating the second graph
import time
from dash.dependencies import Input, Output
from dash import html
from dash import dcc
import dash
import plotly.express as px
from retry_requests import retry
import requests_cache
import openmeteo_requests
from geopy.geocoders import Nominatim
from dash.dependencies import Input, Output, State
import pandas as pd
from threading import Thread
import os
from sqlalchemy import create_engine, text, inspect, Table
import geopandas as gpd
import folium
from meteo_api import get_data

ATTRIBUTES = ['AreaUnderCultivation_1', 'HarvestedArea_2',
              'GrossYieldPerHa_3', 'GrossYieldTotal_4']
ATTRIBUTE_LABEL = ['Area Under Cultivation',
                   'Harvested Area', 'Gross Yield Per Ha', 'Gross Yield Total']
ATTRIBUTE_COLOR = ['Reds', 'Greens', 'Blues', 'YlGnBu']


def log_action(service_name, message):
    try:
        service_indicator = service_name.upper()[0]
    except:
        service_indicator = "X"

    TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f" [{TIMESTAMP}] - [{service_indicator}] {message}")


SERVER_SERVICE_NAME = os.getenv('SERVER_SERVICE_NAME', 'server')
INDICATOR = SERVER_SERVICE_NAME.upper()[0]
SERVER_QUEUE = os.getenv('SERVER_QUEUE', 'server_queue')
cbs_arable_crops, cbs_years, cbs_municipal_boundaries, CBS, MAP_DATA = None, None, None, None, None
engine = create_engine(
    "postgresql://student:infomdss@database:5432/dashboard")

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.8.1/font/bootstrap-icons.min.css"
]

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                title="Agriculture Dashboard")
# Define the style dictionary
style = {
    'width': '33%',
    'maxWidth': '400px',
    'margin': '0 auto'
}

FEATURE_NAMES = None

with engine.connect() as connection:
    # QCL is in quotes because of case sensitivity
    result = connection.execute(
        text('SELECT * FROM "climate_data_attributes"'))
    data = result.fetchall()
    columns = result.keys()
    df = pd.DataFrame(data, columns=result.keys())
    FEATURE_NAMES = dict(zip(df['attribute_code'], df['description']))


def load_and_prepare_data(ch, method, properties, body):
    global current_date, current_date_or, FAOSTAT
    global yearly_average_merged_data, yearly_totals
    global average_growth_rate, mean_value, geom_mean_value
    global min_value, min_year, max_value, max_year, top_products
    global top_5_products, merged_df, correlations, top_5_correlations
    global cbs_arable_crops, cbs_years, cbs_municipal_boundaries, CBS

    if body is not None:
        try:
            log_action(SERVER_SERVICE_NAME, f"Received {body.decode()}")
        except Exception as e:
            pass

    # LOAD CROP DATA
    # FAOSTAT = pd.read_csv("/data/FAOSTAT_nozer.csv")
    FAOSTAT = None
    CBS = None
    yearly_average_merged_data = None

    with engine.connect() as connection:
        # QCL is in quotes because of case sensitivity
        result = connection.execute(text('SELECT * FROM "QCL"'))
        data = result.fetchall()
        columns = result.keys()
        FAOSTAT = pd.DataFrame(data, columns=columns)

    with engine.connect() as connection:
        # CBS is in quotes because of case sensitivity
        result = connection.execute(text('SELECT * FROM "CBS"'))
        data = result.fetchall()
        columns = result.keys()
        CBS = pd.DataFrame(data, columns=columns)

    log_action(SERVER_SERVICE_NAME,
               f"Loaded {FAOSTAT.shape[0]} rows from FAOSTAT")
    log_action(SERVER_SERVICE_NAME,
               f"Loaded {CBS.shape[0]} rows from CBS")

    # LOAD WEATHER DATA
    with engine.connect() as connection:
        result = connection.execute(text('SELECT * FROM "Weather"'))
        data = result.fetchall()
        columns = result.keys()
        yearly_average_merged_data = pd.DataFrame(data, columns=columns)
        yearly_average_merged_data = yearly_average_merged_data.drop(
            'index', axis=1)

    log_action(SERVER_SERVICE_NAME,
               f"Loaded {yearly_average_merged_data.shape[0]} rows from Weather")

    # LOAD WEATHER DATA
    # yearly_average_merged_data = pd.read_csv(
    #     "/data/final_yearly_merged_data.csv")

    # with engine.connect() as connection:
    #     result = connection.execute(text("SELECT * FROM Weather"))
    #     data = result.fetchall()
    #     columns = result.keys()
    #     FAOSTAT = pd.DataFrame(data, columns=columns)

    # CREAT THE DATASET WITH THE TOTAL CROP PER YEAR
    yearly_totals = FAOSTAT.groupby('Year')['Value'].sum().reset_index()

    # Calculate and print the average growth rate
    average_growth_rate = calculate_average_growth_rate(yearly_totals)

    # CALCULATE STATISTICS FOR TOTAL CROP YIELD THROUGH THE YEARS #################
    # 1. Calculate Mean Value
    mean_value = yearly_totals['Value'].mean()

    # 2. Calculate Geometric Mean
    geom_mean_value = average_growth_rate

    # 3. Calculate Min Value and the Year it Occurs
    min_value = yearly_totals['Value'].min()
    min_year = yearly_totals['Year'][yearly_totals['Value'].idxmin()]

    # 4. Calculate Max Value and the Year it Occurs
    max_value = yearly_totals['Value'].max()
    max_year = yearly_totals['Year'][yearly_totals['Value'].idxmax()]

    # CALCULATE CORRELATION BETWEEN WEATHER ATTRIBUTES AND CROP ####################
    # Calculate total production by product
    top_products = FAOSTAT.groupby('Item')['Value'].sum().reset_index()

    # Sort the products by value in descending order and get the top 5
    top_5_products = top_products.nlargest(
        5, 'Value').sort_values(by='Value', ascending=False)

    # Merge DataFrames on Year
    merged_df = pd.merge(yearly_average_merged_data, yearly_totals, on='Year')

    # Calculate Pearson correlations
    correlations = {}
    for column in merged_df.columns:
        if column not in ['Year', 'Value']:
            correlation_coefficient, _ = pearsonr(
                merged_df[column], merged_df['Value'])
            correlations[column] = correlation_coefficient

    # Get the top 5 most correlated weather attributes
    top_5_correlations = pd.Series(correlations).nlargest(5).reset_index()
    top_5_correlations.columns = ['Attribute', 'Correlation']

    # Convert correlation to percentage
    top_5_correlations['Correlation'] = (
        top_5_correlations['Correlation'] * 100).round(2)

    current_date_or = date.today()
    # Get current date - 1 day  i cases the data source isnt updated
    current_date = date.today() - timedelta(days=1)

    # Prepate CBS data
    # Get unique values for dropdowns
    cbs_arable_crops = CBS['ArableCrops'].unique()
    cbs_years = CBS['Periods'].unique()

    # Load geodata
    geodata_url = 'provincie_2024.geojson'
    cbs_municipal_boundaries = gpd.read_file(geodata_url)


def calculate_average_growth_rate(yearly_totals):
    """
    Calculates the average growth rate using the geometric mean.

    Args:
        yearly_totals: DataFrame with 'Year' and 'Value' columns.

    Returns:
        The average growth rate.
    """

    # Calculate yearly growth rates
    yearly_growth_rates = yearly_totals['Value'].pct_change().dropna() + 1

    # Calculate the geometric mean of growth rates
    average_growth_rate = gmean(yearly_growth_rates) - 1

    return average_growth_rate


current_date_or = date.today()
# Get current date - 1 day  i cases the data source isnt updated
current_date = date.today() - timedelta(days=1)

# GENERAL STATISTIC INFO FOR TOTAL CROP YIEL ###################################
load_and_prepare_data(None, None, None, None)


def create_cards():
    return html.Div(
        style={'display': 'flex', 'flexWrap': 'nowrap',
               'overflowX': 'auto', 'justifyContent': 'center'},  # Flexbox settings
        children=[
            dbc.Card(
                dbc.CardBody([
                    html.H5("Average Crop Production", className="card-title"),
                    html.P(f"{mean_value:.2f}", className="card-text")
                ]),
                # Adjust width as needed
                style={'backgroundColor': '#f8f9fa',
                       'boxShadow': '0 4px 8px rgba(0,0,0,0.2)', 'borderRadius': '10px', 'padding': '20px', 'flex': '0 0 200px', 'marginRight': '10px'}
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H5("Average Growth Rate of Crop yield(1961-2022)",
                            className="card-title"),
                    html.P(f"{geom_mean_value*100:.2f}%",
                           className="card-text")
                ]),
                style={'backgroundColor': '#f8f9fa',
                       'boxShadow': '0 4px 8px rgba(0,0,0,0.2)', 'borderRadius': '10px', 'padding': '20px', 'flex': '0 0 200px', 'marginRight': '10px'}
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H5("Minimum Crop production (Year)",
                            className="card-title"),
                    html.P(f"{min_value:.2f} ({min_year})",
                           className="card-text")
                ]),
                style={'backgroundColor': '#f8f9fa',
                       'boxShadow': '0 4px 8px rgba(0,0,0,0.2)', 'borderRadius': '10px', 'padding': '20px', 'flex': '0 0 200px', 'marginRight': '10px'}
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H5("Maximum crop production (Year)",
                            className="card-title"),
                    html.P(f"{max_value:.2f} ({max_year})",
                           className="card-text")
                ]),
                style={'backgroundColor': '#f8f9fa',
                       'boxShadow': '0 4px 8px rgba(0,0,0,0.2)', 'borderRadius': '10px', 'padding': '20px', 'flex': '0 0 200px', 'marginRight': '10px'}
            ),
        ]
    )

# Function to create a circular button and modal


def create_circular_modal(id_prefix, message):
    return html.Div(
        [
            dbc.Button(
                # Bootstrap icon class for question mark
                html.I(className="bi bi-question-circle"),
                id=f"{id_prefix}-open",  # Dynamic id for the open button
                color="primary",
                n_clicks=0,
                style={"borderRadius": "50%", "width": "50px",
                       "height": "50px", "padding": "10px", "textAlign": "center"}
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(
                        "Help")),
                    dbc.ModalBody(id=f"{id_prefix}-body", children=message),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close", id=f"{id_prefix}-close", className="ms-auto", n_clicks=0)
                    ),
                ],
                id=f"{id_prefix}-modal",  # Dynamic id for the modal
                is_open=False,
                centered=True,
            ),
        ]
    )


app.layout = dbc.Container(
    fluid=True,
    children=[    # define the components shown in the app GUI


        # html.Div([
        #     dcc.DatePickerRange(
        #         id='date-picker-range',
        #         min_date_allowed=date(2001, 9, 12),  # Minimum allowed date
        #         max_date_allowed=current_date_or,      # Maximum allowed date (current date)
        #         initial_visible_month=current_date,
        #         start_date=date(2023, 9, 12).strftime('%Y-%m-%d'),  # Default start_date
        #         end_date=current_date.strftime('%Y-%m-%d'),        # Default end_date
        #         style=style  # Apply the style
        #     )
        # ], style={'padding': 10}),

        # CARD PLOT GNERAL STATISTICS FOR TOTAL CROP YIELD THROUGH YEARS ############
        dbc.Container([
            html.H1("Statistics Summary", className="text-center mb-4"),
            create_cards()  # Insert cards here
        ], fluid=True),

        # MAP OF NETHERLANDS #######################################################
        html.Div(className="container-fluid", children=[
            dcc.Store(id='selected-year'),  # Store to hold province name
            dcc.Store(id='province-name'),  # Store to hold province name

            html.Div(className="container mt-5 d-flex justify-content-center  align-items-center", style={'display': 'flex', 'alignItems': 'center'},
                 children=[html.H1(children="Province Information", id="map-header-name", style={"margin": "20px", "lineHeight": "50px"}),
                           create_circular_modal("modal2", "Click on a province to see the weather details for that year."),]),
            html.Div(id="map-tool-bar",
                     style={
                         'display': 'flex',
                         'justifyContent': 'center',  # Center horizontally
                         'alignItems': 'center',  # Center vertically if needed
                         'gap': '10px',  # Reduced space between dropdowns
                         'padding': '20px'  # Optional padding around the container
                     },
                     children=[
                         dcc.Dropdown(
                             id='crop-dropdown',
                             options=[{'label': crop, 'value': crop}
                                      for crop in cbs_arable_crops],
                             value=cbs_arable_crops[0],  # Default value
                             className='dropdown-container',
                             clearable=False,
                             searchable=True,
                             style={
                                 'width': '400px',  # Increased width for better visibility
                                 'fontSize': '16px',  # Increase font size for better readability
                                 'border': '1px solid #ccc',  # Light border
                                 'backgroundColor': '#f9f9f9',  # Light background color
                                 'color': '#333',  # Text color
                             },
                             optionHeight=60

                         ),
                         dcc.Dropdown(
                             id='year-dropdown',
                             options=[{'label': str(year), 'value': year}
                                      for year in cbs_years],
                             value=cbs_years[-1],  # Default value
                             className='dropdown-container',
                             clearable=False,
                             searchable=True,
                             style={
                                 'width': '400px',  # Increased width for better visibility
                                 'fontSize': '16px',  # Increase font size for better readability
                                 'border': '1px solid #ccc',  # Light border
                                 'backgroundColor': '#f9f9f9',  # Light background color
                                 'color': '#333',  # Text color
                             },
                             optionHeight=60

                         ),
                         dcc.Dropdown(
                             id='attribute-dropdown',
                             options=[{'label': ATTRIBUTE_LABEL[index], 'value': index}
                                      for index, attr in enumerate(ATTRIBUTES)],
                             value=0,  # Default value
                             className='dropdown-container',
                             clearable=False,
                             searchable=True,
                             style={
                                 'width': '400px',  # Increased width for better visibility
                                 'fontSize': '16px',  # Increase font size for better readability
                                 'border': '1px solid #ccc',  # Light border
                                 'backgroundColor': '#f9f9f9',  # Light background color
                                 'color': '#333',  # Text color
                             },
                             optionHeight=60

                         ),
                     ]
                     ),
            html.Div(
                className='d-flex justify-content-center mx-auto',  # Centering classes
                children=[
                    # Map container with a set width
                    html.Div(id='map-container',
                             style={'width': '80%'}, children=[dcc.Graph(id='choropleth-map'),])
                ]
            ),
            html.Div(id="year-weather-data", style={'display': 'none'}, children=[
                dbc.Row(style={'justify-content': 'center'}, children=[
                    dbc.Col(
                        dcc.Dropdown(
                            id='feature-dropdown',
                            options=[
                                {'label': 'Minimum Temperature',
                                 'value': 'temperature_2m_min'},
                                {'label': 'Mean Temperature',
                                 'value': 'temperature_2m_mean'},
                                {'label': 'Rain Sum', 'value': 'rain_sum'},
                                {'label': 'Max Wind Speed',
                                    'value': 'wind_speed_10m_max'},
                                {'label': 'Max Wind Gusts',
                                    'value': 'wind_gusts_10m_max'},
                            ],
                            value='temperature_2m_mean',  # Default value
                            clearable=False,
                        ),
                        # Adjust the width as needed (1-12)
                        width=6,
                    ),
                    dbc.Col(
                        html.Button("Back to Map", style={
                            'display': 'block'}, id="back-button-map", n_clicks=0),
                        # Adjust the width as needed (1-12)
                        width=2,
                        # Optional: align button to the right
                        style={
                            'textAlign': 'right'}
                    ),
                ]),
                dcc.Graph(
                    id='feature-graph'),
            ])
        ]),

        # WEATHER AND CROP DATA VISUALIZATION #######################################
        html.H1(children="Weather and Crop Data Visualization"),
        html.Div(
            style={
                'display': 'flex',
                'justifyContent': 'center',  # Center horizontally
                'alignItems': 'center',  # Center vertically if needed
                'gap': '10px',  # Reduced space between dropdowns
                'padding': '20px'  # Optional padding around the container
            },
            children=[
                dcc.Dropdown(
                    id='faostat-item',
                    options=[{'label': item, 'value': item}
                             for item in FAOSTAT['Item'].unique()],
                    # Default value (first Item)
                    value=FAOSTAT['Item'].unique()[2],
                    className='dropdown-container',
                    clearable=False,
                    searchable=True,
                    style={
                        'width': '400px',  # Increased width for better visibility
                        'fontSize': '16px',  # Increase font size for better readability
                        'border': '1px solid #ccc',  # Light border
                        'backgroundColor': '#f9f9f9',  # Light background color
                        'color': '#333',  # Text color
                    },
                    optionHeight=60

                ),
                dcc.Dropdown(
                    id='yearly-data-feature',
                    options=[{'label': FEATURE_NAMES[col], 'value': col}
                             for col in yearly_average_merged_data.columns if col != 'Year'],
                    # Default value (first column after 'Year')
                    value=yearly_average_merged_data.columns[1],
                    className='dropdown-container',
                    clearable=False,
                    searchable=True,
                    style={
                        'width': '400px',  # Increased width for better visibility
                        'fontSize': '16px',  # Increase font size for better readability
                        'border': '1px solid #ccc',  # Light border
                        'backgroundColor': '#f9f9f9',  # Light background color
                        'color': '#333',  # Text color
                    },
                    optionHeight=60

                ),
            ]
        ),

        dcc.Graph(id='yield-graph'),  # New graph for yield data


        # SCATTER PLOT FOR CORELLATION BETWEEN WEATHER ATTRIBUTES AND CROP YIELD ####
        html.H1(children="Correlation between weather attributes and crop yield"),
        html.Div(
            style={
                'display': 'flex',
                'justifyContent': 'center',  # Center horizontally
                'alignItems': 'center',  # Center vertically if needed
                'gap': '10px',  # Reduced space between dropdowns
                'padding': '20px'  # Optional padding around the container
            },
            children=[
                dcc.Dropdown(
                    id='item-dropdown',
                    options=[{'label': item, 'value': item}
                             for item in FAOSTAT['Item'].unique()],
                    value='Mushrooms and truffles',  # Default value
                    className='dropdown-container',
                    clearable=False,
                    searchable=True,
                    style={
                        'width': '400px',  # Increased width for better visibility
                        'fontSize': '16px',  # Increase font size for better readability
                        'border': '1px solid #ccc',  # Light border
                        'backgroundColor': '#f9f9f9',  # Light background color
                        'color': '#333',  # Text color
                    },
                    optionHeight=60

                ),
                dcc.Dropdown(
                    id='weather-attribute-dropdown',
                    options=[{'label': FEATURE_NAMES[col], 'value': col}
                             for col in yearly_average_merged_data.columns if col != 'Year'],
                    value="Mean 2m temperature",  # Default value
                    className='dropdown-container',
                    clearable=False,
                    searchable=True,
                    style={
                        'width': '400px',  # Increased width for better visibility
                        'fontSize': '16px',  # Increase font size for better readability
                        'border': '1px solid #ccc',  # Light border
                        'backgroundColor': '#f9f9f9',  # Light background color
                        'color': '#333',  # Text color
                    },
                    optionHeight=60

                )
            ]
        ),
        # add the gauge meters and put them in the same line with the scatter plot
        html.Div(
            className='row',
            style={'display': 'flex', 'align-items': 'center',
                   'justify-content': 'space-between', 'flex-wrap': 'wrap'},  # Ensure flexible layout
            children=[
                # Scatter plot
                # Make the graph flexible and responsive
                html.Div(
                    className='col-xl-7 col-lg-7 col-md-7 mb-7',  # Responsive columns
                    children=[
                        dcc.Graph(id='scatter-plot',
                                  style={'flex': '1 1 60%', 'min-width': '300px'}),
                    ]
                ),


                # Gauge meters
                html.Div(
                    className='col-xl-5 col-lg-5 col-md-5',
                    style={'display': 'flex', 'flex-direction': 'column', 'flex': '1 1 30%',
                           'min-width': '250px', 'margin-left': '20px'},  # Flexible layout for the gauges
                    children=[
                        daq.Gauge(
                            id='correlation-gauge',
                            label="Correlation",
                            value=0,  # Initial value, will be updated
                            max=1,
                            min=-1,
                            color={"gradient": True, "ranges": {
                                "green": [0.8, 1], "yellow": [0.5, 0.8], "red": [-1, 0.5]}}
                        ),
                        daq.Gauge(
                            id='p-value-gauge',
                            label="Trustability",
                            value=0,  # Initial value, will be updated
                            max=1,
                            color={"gradient": True, "ranges": {
                                "green": [0.95, 1], "yellow": [0.8, 0.95], "red": [0, 0.8]}}
                        ),
                    ]
                ),
            ]
        ),


        # TOTAL CROP YIELD THROUGH THE YEARS ########################################

        html.Div(className="container mt-5 d-flex justify-content-center  align-items-center", style={'display': 'flex', 'alignItems': 'center'},
                 children=[html.H1(children="Total Crop yield through the years", style={"margin": "20px", "lineHeight": "50px"}),
                           create_circular_modal("modal1", "Click on a point in the line graph to see the crop distribution for that year."),]),

        # daq.ToggleSwitch(
        #     id='start-tutorial',
        #     label='Help',
        #     labelPosition='bottom',
        #     value=False,
        #     color='green'
        # ),
        # Store for current tutorial step
        dcc.Store(id='current-step-store', data={'step': 0}),

        # Overlay for tutorial instructions
        html.Div(id='tutorial-overlay', style={'display': 'none', 'position': 'absolute',
                                               'background-color': 'rgba(0, 0, 0, 0.5)',
                                               'color': 'white',
                                               'padding': '20px',
                                               'border-radius': '5px',
                                               'z-index': '1000'}),

        # Line graph is visible initially
        dcc.Graph(id='line-graph', style={'display': 'block'}),
        # Pie chart hidden initially
        dcc.Graph(id='pie-chart', style={'display': 'none', 'width': '100%'}),
        html.Button('Back to Line Chart', id='back-button',
                    style={'display': 'none'}),  # Back button hidden initially

        # TOP MOST PRODUCED CROPS IN NETHERLANDS AND TOP MOST CORRELATE WEATHER ATTRIBUTES ##################################
        html.Div(style={
            'padding': '20px',
            'fontFamily': 'Arial, sans-serif',
            'color': '#333',
            'display': 'flex',
            'justifyContent': 'space-between'  # Align children in a row
        }, children=[
            # Left box for top 5 most produced crops
            html.Div(style={
                'border': '1px solid #ccc',
                'borderRadius': '8px',
                'padding': '20px',
                'margin': '0',
                'width': '48%',  # Adjusted width to fit side by side
                'backgroundColor': '#ffffff',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
            }, children=[
                html.H1("Top 5 Most Produced Crops in the Netherlands",
                        style={'textAlign': 'left', 'color': '#4a4a4a'}),
                *[html.Div(style={'margin': '10px 0', 'padding': '10px', 'backgroundColor': '#f1f1f1', 'borderRadius': '5px'},
                           children=[
                    html.H4(f"{i + 1}. {row.Item}",
                            style={'margin': '0', 'color': '#333'}),
                    html.P(f"Quantity in tons: {row.Value}", style={
                        'margin': '0', 'color': '#555'})
                ])
                    for i, row in enumerate(top_5_products.itertuples(index=False))]  # Use enumerate for correct numbering
            ]),

            # Right box for top 5 most correlated weather attributes
            html.Div(style={
                'border': '1px solid #ccc',
                'borderRadius': '8px',
                'padding': '20px',
                'margin': '0',
                'width': '48%',  # Adjusted width to fit side by side
                'backgroundColor': '#ffffff',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
            }, children=[
                html.H1("Top 5 Most Correlated Weather Attributes with Crop Production", style={
                    'textAlign': 'left', 'color': '#4a4a4a'}),
                *[html.Div(style={'margin': '10px 0', 'padding': '10px', 'backgroundColor': '#f1f1f1', 'borderRadius': '5px'},
                           children=[
                    html.H4(f"{FEATURE_NAMES[row.Attribute]}", style={
                        'margin': '0', 'color': '#333'}),
                    html.P(f"Correlation: {row.Correlation}%", style={
                        'margin': '0', 'color': '#555'})
                ])
                    for row in top_5_correlations.itertuples(index=False)]  # Iterate through the rows
            ])

        ])



    ])
# Tutorial
# @app.callback(
#     Output('tutorial-overlay', 'style'),
#     Output('tutorial-overlay', 'children'),
#     # Output('start-tutorial', 'value'),  # Output for the toggle switch
#     # Input('start-tutorial', 'value'),
#     Input('line-graph', 'clickData'),  # Check for click data on line graph
#     Input('back-button', 'n_clicks'),   # Check for back button clicks
# )
# def update_tutorial(start_tutorial, clickData, n_clicks):
#     # Tutorial steps
#     steps = [
#         {'content': 'Click on a point in the line graph to see the crop distribution for that year.'},
#         {'content': 'This pie chart shows the distribution of crop values for the selected year. Use the back button to return.'}
#     ]

#     # If the tutorial is started
#     if start_tutorial:
#         if n_clicks:  # If back button is clicked
#             # Hide the tutorial and turn off the toggle
#             return {'display': 'none'}, "", False

#         if clickData:  # If a point on the line graph is clicked
#             # Show pie chart tutorial step
#             return {'display': 'block'}, steps[1]['content'], True

#         # If no interaction yet, show the first step
#         # Show line graph tutorial step
#         return {'display': 'block'}, steps[0]['content'], True

#     # If the tutorial is not active and back button is not clicked, ensure to reset
#     return {'display': 'none'}, "", False  # Reset tutorial when not active


# Tutorial


# BAR CHART OF WEATHER ATTRIBUTES AND CROP YIELD#################################
@app.callback(
    Output('yield-graph', 'figure'),
    [Input('yearly-data-feature', 'value'),
     Input('faostat-item', 'value')]
)
def update_yield_graph(selected_feature, selected_item):
    filtered_faostat = FAOSTAT[FAOSTAT['Item'] == selected_item]

    fig = go.Figure()

    fig.add_trace(go.Bar(x=yearly_average_merged_data['Year'],
                         y=yearly_average_merged_data[selected_feature],
                         name=selected_feature,
                         offsetgroup=0))  # Assign to first offset group

    fig.add_trace(go.Bar(x=filtered_faostat['Year'],
                         y=filtered_faostat['Value'],
                         name=selected_item,
                         offsetgroup=1,   # Assign to second offset group
                         yaxis='y2'))  # Assign to secondary y-axis

    fig.update_layout(title=f"{FEATURE_NAMES[selected_feature]} and {selected_item} over Time",
                      xaxis_title="Year",
                      yaxis_title=FEATURE_NAMES[selected_feature],
                      yaxis2=dict(title=selected_item,
                                  overlaying='y', side='right'),
                      barmode='group')  # Set barmode to 'group'

    return fig

# SCATTER PLOT FOR CORRELATION OF CROP AND WEATHER ATRIBUTES ####################


@app.callback(
    Output('scatter-plot', 'figure'),
    Output('correlation-gauge', 'value'),  # Output for correlation gauge
    Output('p-value-gauge', 'value'),    # Output for p-value gauge
    [Input('item-dropdown', 'value'),
     Input('weather-attribute-dropdown', 'value')]
)
def update_scatter_plot(item_name, weather_column):
    value_column = 'Value'

    # Filter FAOSTAT data for the specified item
    item_faostat_data = FAOSTAT[FAOSTAT['Item'] == item_name]

    # Merge the DataFrames based on 'Year'
    merged_data = pd.merge(yearly_average_merged_data, item_faostat_data[[
                           'Year', value_column]], on='Year', how='inner')

    # Extract data for plotting and scaling
    x = merged_data[value_column].values.reshape(-1, 1)
    y = merged_data[weather_column].values.reshape(-1, 1)

    # Calculate Pearson correlation BEFORE scaling
    correlation, p_value = pearsonr(x.flatten(), y.flatten())

    # MinMax scaling
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y)

    # Create scatter plot using plotly.express
    fig = px.scatter(
        x=x_scaled.flatten(),
        y=y_scaled.flatten(),
        labels={
            'x': f"{item_name} Production ({value_column}) - Scaled", 'y': f"{FEATURE_NAMES[weather_column]} - Scaled"},
        title=f"Scatter Plot: {item_name} Production vs. {FEATURE_NAMES[weather_column]}",
        # width=700,  # Set the width
        height=700
    )

    fig.update_layout(
        title={
            # Break into two lines if too long
            'text': f"Scatter Plot: {item_name} Production<br>vs. {FEATURE_NAMES[weather_column]}",
            'y': 0.95,  # Adjust vertical positioning
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 18
            }
        },
    )

    # Add correlation annotation to the plot
    fig.add_annotation(
        x=0.5,
        y=0.95,
        text=f"Correlation: {correlation:.2f}",

        showarrow=False,
        font=dict(size=14)
    ),
    fig.add_annotation(
        x=0.5,
        y=0.90,  # Adjust y position for the p-value annotation
        text=f"P-value: {p_value:.10f}",  # Format p-value to 3 decimal places
        showarrow=False,
        font=dict(size=14)
    )

    return fig, correlation, 1-p_value


# LINE GRAPH FOR TOTAL CROP YIELD THROUGH YEARS #################################
@app.callback(
    Output('line-graph', 'figure'),
    Input('line-graph', 'id')  # dummy input to trigger initial rendering
)
def update_line_graph(_):
    # Create line graph with yearly totals
    fig = go.Figure(go.Scatter(
        x=yearly_totals['Year'], y=yearly_totals['Value'], mode='lines+markers'))
    fig.update_layout(xaxis_title='Year', yaxis_title='Total Value')
    return fig

# DRILL DOW PIE CHART FOR CROP DISTRIBUTION #####


@app.callback(
    Output('line-graph', 'style'),
    Output('pie-chart', 'figure'),
    Output('pie-chart', 'style'),
    Output('back-button', 'style'),
    Output('modal1-body', 'children'),
    Input('line-graph', 'clickData'),
    Input('back-button', 'n_clicks'),
    State('line-graph', 'style'),
    State('current-step-store', 'data')  # Get current tutorial step
)
def toggle_pie_chart(clickData, n_clicks, line_graph_style, current_step_data):
    if clickData and line_graph_style['display'] == 'block':
        clicked_year = clickData['points'][0]['x']
        year_data = FAOSTAT[FAOSTAT['Year'] == clicked_year]
        total_value = year_data['Value'].sum()

        pie_data = go.Pie(
            labels=year_data['Item'], values=year_data['Value'], hole=0.3)
        pie_fig = go.Figure(data=[pie_data])
        pie_fig.update_layout(
            title=f'Distribution for {clicked_year}',
            height=1000,
            width=1400,
            autosize=True,
            margin=dict(l=50, r=50, t=100, b=50),
            legend=dict(x=1.05, y=0.5, orientation='v'),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        pie_fig.add_annotation(
            text=f'Total:<br>{total_value}',
            font_size=15,
            showarrow=False,
            x=0.5,
            y=0.5,
            align='center',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )

        # Show pie chart, hide line chart, and show the "Back" button
        return {'display': 'none'}, pie_fig, {'display': 'block'}, {'display': 'block'}, "This pie chart shows the distribution of crop values for the selected year. Use the back button to return."

    if n_clicks:
        return {'display': 'block'}, {}, {'display': 'none'}, {'display': 'none'}, "Click on a point in the line graph to see the crop distribution for that year."

    return {'display': 'block'}, {}, {'display': 'none'}, {'display': 'none'}, "Click on a point in the line graph to see the crop distribution for that year."

# --------------------------------------------------------------------------------


@app.callback(
    Output('feature-graph', 'figure'),
    Input('feature-dropdown', 'value'),
    Input('selected-year', 'data'),
    Input('province-name', 'data'),
)
def plot_weather(selected_feature, year, province):
    # Check if year and province are provided
    if not year or not province:
        return go.Figure().update_layout(
            title="Please select a province and a year",
            template='plotly_white'
        )

    # Retrieve data
    daily_dataframe = get_data(year, province)

    # Check if the selected feature exists in the dataframe
    if selected_feature not in daily_dataframe.columns:
        return go.Figure().update_layout(
            title="Data not available for selected feature",
            template='plotly_white'
        )

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_dataframe['date'],
        y=daily_dataframe[selected_feature],
        mode='lines+markers',
        name=selected_feature.replace('_', ' ').title()
    ))

    # Update the layout
    fig.update_layout(
        title=f'{selected_feature.replace("_", " ").title()} Over Time for {province} in {year}',
        xaxis_title='Date',
        yaxis_title=selected_feature.replace('_', ' ').title(),
        template='plotly_white'
    )

    return fig


@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('crop-dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('attribute-dropdown', 'value'),
     ]
)
def update_map(selected_crop, selected_year, selected_attribute):
    global CBS, cbs_municipal_boundaries, MAP_DATA
    # Filter the data based on the selected crop and year
    filtered_data = CBS[
        (CBS['ArableCrops'] == selected_crop) &
        (CBS['Periods'] == selected_year) &
        (CBS['Regions'].str.endswith('(PV)'))
    ].copy()

    # Update the name column
    filtered_data.loc[:, 'name'] = filtered_data['Regions'].str.replace(
        r'\s*\(PV\)$', '', regex=True)

    # Merge with geodata
    MAP_DATA = pd.merge(cbs_municipal_boundaries, filtered_data,
                        left_on="statnaam", right_on="name")
    MAP_DATA = MAP_DATA.to_crs(epsg=4326)

    # if MAP_DATA.empty or MAP_DATA[ATTRIBUTES[selected_attribute]].isnull().all():
    #     return html.Div("No data available for the selected crop and year.", style={'textAlign': 'center', 'padding': '20px'})

    fig = go.Figure()

    # Add choropleth map layer
    fig.add_trace(go.Choroplethmapbox(
        geojson=MAP_DATA.geometry.__geo_interface__,  # Use the GeoJSON geometry
        locations=MAP_DATA.index,
        # Use the selected attribute for coloring
        z=MAP_DATA[ATTRIBUTES[selected_attribute]],
        colorscale=ATTRIBUTE_COLOR[selected_attribute],  # Color scale
        # Update the colorbar title
        colorbar_title=ATTRIBUTE_LABEL[selected_attribute],
        marker_opacity=0.7,
        marker_line_width=0,
        hoverinfo='text',  # Specify that you want to display text on hover
        # Show region name
        hovertemplate='<b>Region:</b> %{customdata[0]}<br>' +
                      '<b>Crop:</b> %{customdata[1]}<br>' +
                      '<b>Year:</b> %{customdata[2]}<br>' +
                      f'<b>{ATTRIBUTE_LABEL[selected_attribute]}:</b> %{{z}}<br>' +
                      '<extra></extra>',  # Removes the secondary hover text
        customdata=MAP_DATA[['name', 'ArableCrops', 'Periods']].values
    ))

    nld_lat = 52.2130
    nld_lon = 5.2794
    # Set the mapbox style to OpenStreetMap
    fig.update_layout(
        mapbox_style="open-street-map",  # Using OpenStreetMap style
        mapbox_zoom=5.5,  # Set zoom level
        mapbox_center={"lat": nld_lat, "lon": nld_lon},  # Center of the map
        title_text=f"{ATTRIBUTE_LABEL[selected_attribute]} by Region",
        height=600,  # Set height of the figure
    )

    return fig  # Return the figure as a dcc.Graph component


@app.callback(
    Output('choropleth-map', 'style'),
    Output('year-weather-data', 'style'),
    Output('back-button-map', 'n_clicks'),
    Output('map-tool-bar', 'style'),
    Output('choropleth-map', 'clickData'),
    Output('selected-year', 'data'),
    Output('province-name', 'data'),
    Output('map-header-name', 'children'),
    Output('modal2-body', 'children'),
    Input('choropleth-map', 'clickData'),
    Input('back-button-map', 'n_clicks'),
    Input('year-dropdown', 'value'),
)
def toggle_views(clickData, n_clicks, year):
    global CBS, MAP_DATA
    MAP_VIEW = "Province Information"
    PLOT_VIEW = "Weather Data per Year"
    active_toolbar_style = {
        'display': 'flex',
        'justifyContent': 'center',  # Center horizontally
        'alignItems': 'center',  # Center vertically if needed
        'gap': '10px',  # Reduced space between dropdowns
        'padding': '20px'  # Optional padding around the container
    }
    hidden_toolbar_style = {
        'display': 'none',
        'justifyContent': 'center',  # Center horizontally
        'alignItems': 'center',  # Center vertically if needed
        'gap': '10px',  # Reduced space between dropdowns
        'padding': '20px'  # Optional padding around the container
    }

    if n_clicks:
        return {'display': 'block'}, {'display': 'none'}, 0, active_toolbar_style, None, year, None, MAP_VIEW, "Click on a province to see the weather details for that year."

    if clickData:
        province_index = clickData['points'][0]['location']
        province_name = MAP_DATA.loc[province_index, 'name']
        return {'display': 'none'}, {'display': 'block'}, 0, hidden_toolbar_style, None, year, province_name, PLOT_VIEW, "You can go back to the national map by clicking the back button."

    return {'display': 'block'}, {'display': 'none'}, 0, active_toolbar_style, None, year, None, MAP_VIEW, "Click on a province to see the weather details for that year."

# --------------------------------------------------------------------------------
# Callback to toggle the modal for modal1


@app.callback(
    Output("modal1-modal", "is_open"),
    [Input("modal1-open", "n_clicks"), Input("modal1-close", "n_clicks")],
    [State("modal1-modal", "is_open")],
)
def toggle_modal1(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Callback to toggle the modal for modal2


@app.callback(
    Output("modal2-modal", "is_open"),
    [Input("modal2-open", "n_clicks"), Input("modal2-close", "n_clicks")],
    [State("modal2-modal", "is_open")],
)
def toggle_modal2(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


def on_message(channel, method_frame, header_frame, body):
    log_action(SERVER_SERVICE_NAME,
               f"Received message: {body.decode('utf-8')}")
    # Manually acknowledge the message
    channel.basic_ack(delivery_tag=method_frame.delivery_tag)


def on_connected(connection):
    # Open a channel once the connection is established
    connection.channel(on_open_callback=on_channel_open)


def on_channel_open(channel):
    # Declare the queue
    channel.queue_declare(queue=SERVER_QUEUE)
    # Start consuming messages
    channel.basic_consume(SERVER_QUEUE, load_and_prepare_data)


def rabbitmq_consumer():
    parameters = pika.ConnectionParameters(host='rabbitmq', heartbeat=600)
    connection = pika.SelectConnection(
        parameters, on_open_callback=on_connected)

    log_action(SERVER_SERVICE_NAME, "Starting RabbitMQ consumer...")
    # Non-blocking I/O loop
    try:
        connection.ioloop.start()
    except Exception as e:
        log_action(SERVER_SERVICE_NAME, f"Error in consumer: {e}")
        connection.ioloop.stop()


# Run the app
if __name__ == '__main__':

    consumer_thread = Thread(target=rabbitmq_consumer, daemon=True)
    consumer_thread.start()

    app.run_server(debug=True, host='0.0.0.0', port=8050, use_reloader=True)
