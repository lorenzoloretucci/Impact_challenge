import configparser
import json
import os
import random

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backend.path_planning import path_planning
from backend.zone_splitting import kmeans_subdivision
from dash.dependencies import Output, Input
from openrouteservice import client

# from backend.prediction import MakePrediction
# import tensorflow as tf


# Reproducibility
random.seed(123)
np.random.seed(123)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Load configs
curr_dir = os.getcwd()
config_file = os.path.join(curr_dir, 'configs.ini')
config = configparser.ConfigParser()
config.read(config_file)
API_KEY = config['GENERAL'].get('API_KEY')
UPDATE_INTERVAL = config['GENERAL'].getint('UPDATE_INTERVAL')

# APP PROPERTIES
app = dash.Dash(name='UnWaste! FrontEnd')
server = app.server
app.title = "UnWaste! Project"
app.config.suppress_callback_exceptions = True
# OpenRouteService client
clnt = client.Client(key=API_KEY)  # Create client with api key

# SETTINGS AND VARIABLES
# Simulated DBs
garbage_bins = pd.read_csv('./DATABASE/coords_groups.csv')
garbage_trucks = pd.read_csv('DATABASE/trucks_coords.csv')
bins_pred_df = pd.read_csv('./DATABASE/coords_groups.csv')  # copy for different display page
bins_past_state = pd.read_csv("./DATABASE/latest_time_obs.csv")
# Trucks report Dataframe creation 
trucks_report_df = garbage_trucks.copy()
type_report = ['Report', "Issue", "not-specified", "Injury"]
type_name = ['Anil', "Giuliano", "Marco", "Alberto", "Giuliana", "Maria", "Paola", "Ignazio"]
driver_names = [np.random.choice(type_name) for _ in range(len(trucks_report_df))]
truck_reports = [np.random.choice(type_report) if trucks_report_df['available'][i] == 0 else 'ACTIVE' for i in range(len(trucks_report_df))]
trucks_report_df['Drivers'] = driver_names
trucks_report_df['Status'] = truck_reports
del trucks_report_df['available']
trucks_report_df.rename(columns = {'truck_id':'Truck number',
                                   'latitude': 'Latitude',
                                   'longitude': 'Longitude'}, inplace=True)
#  Trucks status Dataframe creation
condition_fuel_mount = ['Empty', 'Full', '50%']
time_of_day = ["morning", "evening", "afternoon"]
truck_status = {'Fuel_L': [random.randrange(1, 80, 1) for i in range(len(trucks_report_df))],
                "mount_mc": [random.randrange(1, 15, 1) for i in range(len(trucks_report_df))],
                "truck_fuel_situation": [random.choice(condition_fuel_mount) for i in range(len(trucks_report_df))],
                "time": [random.choice(time_of_day) for i in range(len(trucks_report_df))]}
truck_status_df = pd.DataFrame(truck_status)
# Waste summary Dataframe 
waste_type = ['PET', 'alluminium', "paper", "glassware", "metalware", "undifferentiated"]
TIME_week = ['Monday', "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Suday"]
waste_summary = {"waste": [random.choice(waste_type) for i in range(len(trucks_report_df))],
                 "total_waste": [random.randrange(1, 648, 1) for i in range(len(trucks_report_df))],
                 "day": [random.choice(TIME_week) for i in range(len(trucks_report_df))]}
waste_summary_df = pd.DataFrame(waste_summary)
wastes = waste_summary_df.waste.unique()
# Variables
START_COORDS = (41.89117549369146, 12.502362854652286)
POSITIONS = garbage_bins[['latitude', 'longitude']].values
GARBAGE_TRUCKS_COORDS = garbage_trucks[['latitude', 'longitude']].values
available_garbage_trucks = garbage_trucks['available'].sum()  # only available garbage trucks
GARBAGE_LABELS = [{'label': 'None', 'value': -1}]
for k in garbage_trucks['truck_id']:
    GARBAGE_LABELS.append({'label': f'Truck #{k + 1}', 'value': str(k)})
SHOW_ROUTES = {i: False for i in range(len(garbage_trucks))}

bins_full = []
paths = {'trucks': [],
         'zone_0': []
        }
bins_pred_df['latest_pred'] = [0. for _ in range(len(POSITIONS))]

# DASH CALLBACKS
# Update the Folium map
@app.callback(Output('map', 'srcDoc'),
              Input('interval-component',
                    'n_intervals'))  # add an input here to load the pathon the map at a user's notice
def update_map(n):
    global START_COORDS, POSITIONS, GARBAGE_TRUCKS_COORDS, SHOW_ROUTES, paths

    rome_map = folium.Map(location=START_COORDS, title="Rome", zoom_start=16, min_zoom=16, max_zoom=18)

    # add not-full garbage bins maps
    for i, p in enumerate(POSITIONS):
        if i not in bins_full:
            folium.Marker(location=[p[0], p[1]],
                          icon=folium.features.CustomIcon("https://i.imgur.com/7Z5nOWb.png",  # empty bins icon
                                                          icon_size=(20, 20)),
                          popup=f'Garbage bin #{int(i)}'
                          ).add_to(rome_map)
        else:
            folium.Marker(location=[p[0], p[1]],
                          icon=folium.features.CustomIcon("https://i.imgur.com/umB38Re.png",  # full bins icon
                                                          icon_size=(20, 20)),
                          popup=f'Garbage bin #{int(i)}'
                          ).add_to(rome_map)

    # draw active trucks
    active_trucks_pos = garbage_trucks.loc[
        garbage_trucks['available'] == 1, ['truck_id', 'latitude', 'longitude']].values
    for pos in active_trucks_pos:
        folium.Marker(location=[pos[1], pos[2]],
                      icon=folium.features.CustomIcon("https://i.imgur.com/qsFNVJj.png",  # active trucks
                                                      icon_size=(35, 35)),
                      popup=f'Garbage truck #{int(pos[0] + 1)}'
                      ).add_to(rome_map)
    # draw inactive trucks
    inactive_trucks_pos = garbage_trucks.loc[
        garbage_trucks['available'] == 0, ['truck_id', 'latitude', 'longitude']].values
    for pos in inactive_trucks_pos:
        folium.Marker(location=[pos[1], pos[2]],
                      icon=folium.features.CustomIcon("https://i.imgur.com/eSyUS0b.png",  # inactive trucks
                                                      icon_size=(35, 35)),
                      popup=f'Garbage truck #{int(pos[0] + 1)}'
                      ).add_to(rome_map)

    # take trucks position, add it to maps
    for truck in garbage_trucks['truck_id'].values:

        # get directions
        if truck in paths['trucks'] and SHOW_ROUTES[
            truck]:  # decomment to use API; TODO: add checkbox to toggle route drawing
            bins_ids = paths['zone_' + str(paths['trucks'].index(truck))]
            coordinates = [[GARBAGE_TRUCKS_COORDS[truck][1], GARBAGE_TRUCKS_COORDS[truck][0]]] + [
                [POSITIONS[idx][1], POSITIONS[idx][0]] for idx in bins_ids]
            route = clnt.directions(coordinates=coordinates,
                                    profile='driving-car',
                                    format='geojson',
                                    preference='fastest',
                                    geometry=True,
                                    geometry_simplify=True)
            # swap lat/long for folium
            points = [[p[1], p[0]] for p in route['features'][0]['geometry']['coordinates']]

            folium.PolyLine(points, color='#3b5998', weight=8, opacity=0.6).add_to(rome_map)

    return rome_map._repr_html_()


# Update which route to show
@app.callback(
    dash.dependencies.Output('ignore-me', 'children'),
    [dash.dependencies.Input('input-on-submit', 'value')])
def update_output(value):
    global SHOW_ROUTES
    truck_n = int(value)
    for k in SHOW_ROUTES.keys():
        SHOW_ROUTES[k] = False
    if truck_n != -1 and truck_n in SHOW_ROUTES.keys():
        SHOW_ROUTES[truck_n] = True
    return ''

@app.callback(
    dash.dependencies.Output('ignore-me2', 'children'),
    dash.dependencies.Input('prediction-submit', 'n_clicks'))
def update_predictions(n_clicks):
    global bins_full, paths
    # Compute predictions at next timestep
    # predictor = MakePrediction(' .')
    if n_clicks > 0:
        # bins_full = otherwise, predictor.prediction()
        bins_full = np.random.choice(np.arange(len(POSITIONS)), size=len(POSITIONS) * 50 // 100, replace=False)
        # update with last prediction
        bins_pred_df['latest_pred'] = [1. if i in bins_full else 0. for i in range(len(POSITIONS))]
        # precompute paths
        clusters, centers = kmeans_subdivision(bins_full, '.', available_garbage_trucks)
        paths = path_planning(clusters, centers, '.')

# Generate / Update the trucks status' chart
@app.callback(
    Output("pie-chart", "figure"),
    [Input("names", "value"),
     Input("values", "value")])
def generate_chart(names, values):
    fig = px.pie(truck_status_df, values=values, names=names, color_discrete_sequence=px.colors.sequential.Jet)
    return fig


# Generate / Update the waste status' chart
@app.callback(
    Output("histo", "figure"),
    [Input("waste", "value")])
def display_color(waste):
    waste_type = waste_summary_df['waste'] == waste
    fig = px.bar(waste_summary_df[waste_type], x='day', y='total_waste')
    return fig


# Update the rows of the single garbage bin table
@app.callback(
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('datatable-interactivity', 'selected_rows')
)
def update_styles(selected_rows):
    return [{
        'if': {'row_index': i},
        'background_color': '#D2F3FF'
    } for i in selected_rows]


# Update the garbage bin filling history / prediction graph
@app.callback(
    Output("line-chart", "figure"),
    [Input('datatable-interactivity', 'selected_rows')])
def update_line_chart(selected_rows):
    fig = go.Figure()
    fig.update_layout(title='Past and predicted garbage bin fullness',
                      yaxis_title='Fullness (%)',
                      xaxis_title='Timesteps (~4 hrs)')
    if len(selected_rows) > 0:
        selected_row = selected_rows[0]
        past_states = bins_past_state.iloc[selected_row].values[1:]
        predicted_state = bins_pred_df.iloc[selected_row]['latest_pred']
        fig.add_trace(go.Scatter(x=[i for i in range(len(past_states))], y=past_states, name=f'Bin #{selected_rows[0]}',
                                 line=dict(color='royalblue', width=4, dash='dash')))

        fig.add_trace(go.Scatter(x=[len(past_states) - 1, len(past_states)], y=[past_states[-1], predicted_state],
                                 name=f'Pred. Bin #{selected_rows[0]}',
                                 line=dict(color='darkolivegreen', width=4, dash='dash')))
    return fig


# Navigation callback
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return home
    elif pathname == "/page-1":
        return BIN
    elif pathname == "/page-2":
        return TRUCKS
    elif pathname == "/page-3":
        return HELP
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


# HTML PAGES
# sidebard
sidebar = html.Div(
    [
        html.Img(src="https://i.imgur.com/jwzOPDb.png", className="Logo"),  # Logo
        html.Hr(className="sidebar_divisor"),
        dbc.Nav(
            [
                dbc.NavLink(html.Img(src="https://i.imgur.com/rCuGj8H.png", width=40, height=40, className="home"),
                            href="/", active="exact", className='nav'),
                # Home
                dbc.NavLink(html.Img(src="https://i.imgur.com/LXXgKM4.png", width=40, height=40, className="bins"),
                            href="/page-1", active="exact", className='nav'),  # Bins
                dbc.NavLink(html.Img(src="https://i.imgur.com/ZOFEbQ9.png", width=40, height=40, className="truks"),
                            href="/page-2", active="exact", className='nav'),
                # Truks
                dbc.NavLink(html.Img(src="https://i.imgur.com/qrIKKYb.png", width=40, height=40, className="info"),
                            href="/page-3", active="exact", className='nav'),
                # info
            ],
            vertical=True,
            pills=True,
            className="navbar"
        ),
    ],
    className='sidebar',
)
# home
home = html.Div(
    # MAIN
    children=[
        # Body1#
        html.Div(children=[
            # Map#
            html.Div([
                html.Iframe(id='map', srcDoc=None, className='inframe_map'),
                # Route selection
                html.Div([
                    html.Div([
                        html.P("Select truck path:", className='name_selector'),
                        html.Div([dcc.Dropdown(id='input-on-submit', options=GARBAGE_LABELS, value='-1', className='nav_map'),
                                html.Div(id='ignore-me', hidden=True)
                                ], style={"width": "100%"})
                    ], className="left-stats1"),
                    html.Div([
                        html.P("Generate prediction:", className='name_selector'),
                        html.Div([html.Button('GENERATE', id='prediction-submit', n_clicks=0, className ='generate_button'),
                                html.Div(id='ignore-me2', hidden=True)
                                ], style={"width": "100%", "margin-top": "10px"})
                    ], className="right-stats1")                   
                ], className="wrapper4")
            ], className="Map"),
            # Report#
            html.Div([
                # reportTitle
                html.H3('Real-Time Reports', className='wintitle'),
                # Table
                dash_table.DataTable(id='table', columns=[{"name": i, "id": i} for i in trucks_report_df.columns],
                                     data=trucks_report_df.to_dict('records')),
            ], className="Report"),
            html.Div([
                dcc.Interval(
                    id='interval-component',
                    interval=UPDATE_INTERVAL,
                    n_intervals=0)
            ]),
        ], className='wrapper1'),
        ## body 2
        html.Div(children=[
            ##left-stats
            html.Div([
                html.Div(children=[
                    html.H3("Trucks Situation"),
                    html.P("Names:", className='name_selector'),
                    dcc.Dropdown(
                        id='names',
                        value='truck_fuel_situation',
                        options=[{'value': x, 'label': x} for x in ['truck_fuel_situation', 'time']],
                        clearable=False
                    ),
                    html.P("Values:", className='name_selector'),
                    dcc.Dropdown(
                        id='values',
                        value='Fuel_L',
                        options=[{'value': x, 'label': x} for x in ['Fuel_L', 'mount_mc']],
                        clearable=False
                    ),
                ], className='nav_r_stats'),
                html.Div(children=[
                    dcc.Graph(id="pie-chart", className='graph',
                              config={"responsive": True, "autosizable": True, "fillFrame": False})

                ], className="graph")
            ], className="left-stats"),
            # right Stats
            html.Div([
                html.Div(children=[
                    html.H3("Waste Situation"),
                    html.P("Waste:", className='name_selector'),
                    dcc.Dropdown(
                        id='waste',
                        value='PET',
                        options=[{'value': x, 'label': x} for x in wastes],
                        clearable=False
                    ),
                ], className="nav_l_stats"),
                html.Div(children=[
                    dcc.Graph(id="histo", className='graph',
                              config={"responsive": True, "autosizable": True, "fillFrame": False}),
                ], className='graph')
            ], className="right-stats")
        ], className="wrapper2"),
    ], className="HTML"
)

# bins
BIN = html.Div( children = [ 
    html.Div( children = [
        html.H1('Bin Prediction'),
        html.P("This page shows at the moment the prediction of bin's state"),
        html.P("In future we will update different windows to manage other bins data....")
    ], className = "trucks_page" ),
    html.Div(children=[
    # Table 1
    html.Div(children=[
         
        html.H3("Bins"),
        html.Div(
            dash_table.DataTable(
                id='datatable-interactivity',
                columns=[
                    {"name": i, "id": i, "selectable": True} for i in bins_pred_df.columns
                ],
                data=bins_pred_df.to_dict('records'),  # TODO: refresh when we generate predictions
                editable=True,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="single",
                row_deletable=False,
                selected_columns=[],
                selected_rows=[0],
                page_action="native",
                page_current=0,
                page_size=10,
            ),
        ),
    ], className='bindata'),
    # Prediction
    html.Div(children=[
        html.H3("Forecast Plot"),
        dcc.Graph(id="line-chart", className='graph_bin',
                  config={"responsive": True, "autosizable": True, "fillFrame": False})
        ], className='graph_bin')
    ], className="wrapper3"
)

]
)
# trucks
TRUCKS = html.Div(children=[
    html.Div(children=[
        html.H1('Trucks overview')
    ]),
    html.P('Here you can find all the informations about the active trucks.'),
    html.Div(children=[
        html.H3("Looks like there's nothing here (yet)"),
        html.P("In the future, here you will find all information and settings regarding the path planning."),
        html.P("For example, you will be able to choose the subset of bins or the number of \"zones\" assigned to the trucks."),
        html.P("You will be able to also explore through various statistical and analysis results about the data you collected.")
    ])
], className= "trucks_page")
# help
HELP = html.Div(children=[
    html.Div(children=[
        html.H1('This is the HELP page')
    ]),
    html.Div(children=[
        html.H3('What is this?'),
        html.P("This is a dashboard where you can plan your entire garbage collection process."),
        html.H3('How can I predict garbage filling?'),
        html.P("You can predict the garbage filling and plan the garbage collection route in your area by clicking the \"Predict\" button under the map."),
        html.H3('Where can I see the trucks?'),
        html.P("Your trucks are all on the map! You can select which one to focus on by selecting its unique identifier in the dropdown menu."),
        html.H3('Where can I see the bin history?'),
        html.P("Our system keeps track of all your bins' history! This can be accessed at the 'Bin overview' page, where you will see both the history of the bins and our predictions."),
        html.H3('Where can I get some support?'),
        html.P("You can ask for support at totally-valid-mail@notatest.com.")
    ])
], className = "Help_page")
# main content
content = html.Div(id="page-content", className="content")

# set the app's layout
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

if __name__ == "__main__":
    app.run_server(debug=True)
