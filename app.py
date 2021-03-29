import dash 
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input
import dash_table
from openrouteservice import client
import json
import configparser
import os
import folium
import plotly.express as px
import random
from backend.path_planning import path_planning
from backend.zone_splitting import kmeans_subdivision
#from backend.prediction import MakePrediction
#import tensorflow as tf

random.seed(123)
np.random.seed(123)

#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

# EXTERNAL SETTINGS
# should be loaded elsewhere and imported here / in a submodule

START_COORDS = (41.89117549369146, 12.502362854652286)
# positions are divided in groups for each garbage-truck
garbage_bins = pd.read_csv('./DATABASE/coords_groups.csv')
POSITIONS = garbage_bins[['latitude','longitude']].values
garbage_trucks = pd.read_csv('DATABASE/trucks_coords.csv')
GARBAGE_TRUCKS = garbage_trucks[['latitude','longitude']].values
available_garbage_trucks = garbage_trucks['available'].sum()  # only available garbage trucks

# predictor = MakePrediction('.')

GARBAGE_LABELS = []
for k in garbage_trucks['truck_id']:
    GARBAGE_LABELS.append({'label': f'Truck #{k + 1}', 'value': str(k)})
SHOW_ROUTES = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}

# precompute paths
#bins_full = predictor.prediction()

bins_full = np.array([0,  1,  2,  3,  4,  5,  6,  7, 12, 14, 19, 21, 22, 23])
clusters, centers = kmeans_subdivision(bins_full, '.', available_garbage_trucks)
paths = path_planning(clusters, centers, '.')

clnt = client.Client(key=API_KEY) # Create client with api key

# theoretically, the paths / markups should be loaded in real time from some service
# in this demo, however, we can just add them from a database of precoumputed positions
@app.callback(Output('map', 'srcDoc'),
              Input('interval-component', 'n_intervals'))  # add an input here to load the pathon the map at a user's notice
def update_map(n):
    global START_COORDS, POSITIONS, GARBAGE_TRUCKS, SHOW_ROUTES, paths

    rome_map = folium.Map(location = START_COORDS, title = "Rome", zoom_start = 16, min_zoom = 16, max_zoom = 18)

    #add not-full garbage bins maps
    for i, p in enumerate(POSITIONS):
        if i not in bins_full:
            folium.Marker(location=[p[0], p[1]],
                        icon = folium.features.CustomIcon( "https://i.imgur.com/LXXgKM4.png",
                                                        icon_size=(20, 20)),
                                                        popup=f'Garbage bin #{int(i)}'
                        ).add_to(rome_map)
        else:
            folium.Marker(location=[p[0], p[1]],
                        icon = folium.features.CustomIcon( "https://i.imgur.com/y8mMdnB.png",
                                                        icon_size=(20, 20)),
                                                        popup=f'Garbage bin #{int(i)}'
                        ).add_to(rome_map)

    # draw active trucks
    active_trucks_pos = garbage_trucks.loc[garbage_trucks['available'] == 1, ['truck_id', 'latitude','longitude']].values
    for pos in active_trucks_pos:
        folium.Marker(location=[pos[1], pos[2]],
                                icon = folium.features.CustomIcon( "https://i.imgur.com/qsFNVJj.png",
                                                                icon_size=(35, 35)),
                                                                popup=f'Garbage truck #{int(pos[0] + 1)}'
                    ).add_to(rome_map)
    # draw inactive trucks
    inactive_trucks_pos = garbage_trucks.loc[garbage_trucks['available'] == 0, ['truck_id', 'latitude','longitude']].values
    for pos in inactive_trucks_pos:
        folium.Marker(location=[pos[1], pos[2]],
                                icon = folium.features.CustomIcon( "https://i.imgur.com/eSyUS0b.png",
                                                                icon_size=(35, 35)),
                                                                popup=f'Garbage truck #{int(pos[0] + 1)}'
                    ).add_to(rome_map)

    #take trucks position, add it to maps
    for truck in garbage_trucks['truck_id'].values:

        # get directions
        if truck in paths['trucks'] and SHOW_ROUTES[truck]:  # decomment to use API; TODO: add checkbox to toggle route drawing
            bins_ids = paths['zone_' + str(paths['trucks'].index(truck))]
            coordinates = [[GARBAGE_TRUCKS[truck][1], GARBAGE_TRUCKS[truck][0]]] + [[POSITIONS[idx][1], POSITIONS[idx][0]] for idx in bins_ids] 
            route = clnt.directions(coordinates=coordinates,
                                        profile='driving-car',
                                        format='geojson',
                                        preference='fastest',
                                        geometry=True,
                                        geometry_simplify=True)
            # swap lat/long for folium
            points = [[p[1], p[0]] for p in route['features'][0]['geometry']['coordinates']]

            folium.PolyLine(points, color='red', weight=10, opacity=0.8).add_to(rome_map)

    return rome_map._repr_html_()

@app.callback(
    dash.dependencies.Output('ignore-me', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    global SHOW_ROUTES
    truck_n = int(value)
    if truck_n in SHOW_ROUTES.keys():
        for k in SHOW_ROUTES.keys():
            SHOW_ROUTES[k] = False
        SHOW_ROUTES[truck_n] = True
    return ''

##### callback for left-stats #####
@app.callback(
    Output("pie-chart", "figure"), 
    [Input("names", "value"), 
     Input("values", "value")])
def generate_chart(names, values):
    fig = px.pie(df_left, values= values, names= names)
    return fig


##### callback for right- stats ###
@app.callback(
    Output("histo", "figure"), 
    [Input("waste", "value")])
def display_color(waste):
    waste_type = df_right['waste'] == waste
    fig = px.bar(df_right[waste_type], x ='day', y = 'total_waste')
    return fig





###############  Dataframe ############################## 
choose_len = 20
choose_len2 = 14

Type_report = ['Report', "Issiue", "NaN", "Injury"]
Type_name = ['Anil',"Giuliano", "Marco", "Alberto", 
                "Giuliana", "Maria", "Paola", "Ignazio"]

data = {"Report_id":[random.randrange(1257,3679,1) for i in range(choose_len2)], 
        "Truck_id":[random.randrange(1,60,1) for i in range(choose_len2)],
         "Type":[random.choice(Type_report) for i in range(choose_len2)],
         "Operator":[random.choice(Type_name) for i in range(choose_len2)] }

df = pd.DataFrame(data)


#####  Trucks Dataframe ###
condition_fuel_mount =['Empty', 'Full', '50%']
TIME = ["morning", "evening", "afternoon"]

data_left = {'Fuel_L':[random.randrange(1,80,1) for i in range(choose_len)],
             "mount_mc": [random.randrange(1,15,1) for i in range(choose_len)],
             "truck_fuel_situation": [random.choice(condition_fuel_mount) for i in range(choose_len)],
             "time": [random.choice(TIME) for i in range(choose_len)]}

df_left = pd.DataFrame(data_left)

####### Waste dataframe 
waste_type = ['PET', 'alluminium', "paper", "glassware", "metalware", "undifferentiated"]
TIME_week = ['Monday',"Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Suday"]

data_right = {"waste": [random.choice(waste_type) for i in range(choose_len)],
            "total_waste":[random.randrange(1,648,1) for i in range(choose_len)],
            "day": [random.choice(TIME_week) for i in range(choose_len)]}


df_right = pd.DataFrame(data_right)
wastes = df_right.waste.unique()
###########################################################


app.layout = html.Div(
    #MAIN
            children = [    
                #header#
                        html.Div(children = [
                                            html.H1("UnWaste! | Demo Dashboard", className = "header-title"),
                                            #   html.P('Demo dashboard', className = 'header-description')
                                            ],
                                className = 'header'
                                ),
                #Body1#
                        html.Div(children = [
                                            #Map#
                                            html.Div([
                                                        #html.H3('Path Map',  className = 'wintitle'),
                                                        html.Iframe(id = 'map', srcDoc = None, className = 'inframe_map' ),
                                            #Button div
                                                        html.Div([dcc.Dropdown(id='input-on-submit', options = GARBAGE_LABELS, value='0', className = 'nav_map' ),
                                                                    html.Button('Submit', id='submit-val', n_clicks=0, className = 'Button_map' ),
                                                                    html.Div(id='ignore-me', hidden=True)
                                                                 ])
                                                     ],
                                                        
                                                    className = "Map"),
                                        
                                            #Report#
                                            html.Div([
                                                    # reportTitle
                                                    #html.H3('Real-Time Reports',  className = 'wintitle'),
                                                     #Table
                                                    dash_table.DataTable(id='table',columns=[{"name": i, "id": i} for i in df.columns],data=df.to_dict('records')),
                                                    
                                                    ],
                                                    className="Report"
                                                    ),
                                            html.Div([
                                                dcc.Interval(
                                                id='interval-component',
                                                interval=UPDATE_INTERVAL,
                                                n_intervals=0)
                                                    ]),

                    
                                            
                                            ],
                                            
                                        className = 'wrapper1'),
                        ## body 2
                        html.Div(children = [
                                            ##left-stats
                                            html.Div([ 
                                                    #html.H3('Real-time Trucks stats', className = 'wintitle'),
                                                    html.P("Names:", className = 'name_selector'),
                                                    dcc.Dropdown(
                                                                id='names', 
                                                                value='truck_fuel_situation', 
                                                                options=[{'value': x, 'label': x} for x in ['truck_fuel_situation', 'time']],
                                                                clearable=False
                                                                ),
                                                    html.P("Values:", className = 'name_selector'),
                                                    dcc.Dropdown(
                                                                id='values', 
                                                                value='Fuel_L', 
                                                                options=[{'value': x, 'label': x} for x in ['Fuel_L', 'mount_mc']],
                                                                clearable=False
                                                                ),
                                                    
                                                    dcc.Graph(id="pie-chart", className= 'graph'),
                                                  
                                                    ],
                                                    className = "left-stats"),
                                            #right Stats
                                            html.Div([
                                                    #html.H3('Real-Time bin stats',  className = 'wintitle'),
                                        
                                                    html.P("Waste:", className = 'name_selector'),
                                                    dcc.Dropdown(
                                                                id='waste', 
                                                                value='PET', 
                                                                options=[{'value': x, 'label': x} for x in wastes],
                                                                clearable=False
                                                                ),

                                                       dcc.Graph(id="histo", className= 'graph'),
                                                            

                                                    ],
                                                    className="right-stats")
                                            

                            



                                            ], 
                                            className = "wrapper2"),
                                  
                        ],
className="HTML"
)





if __name__ == "__main__":
    app.run_server(debug=True)