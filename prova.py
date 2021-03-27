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
from backend.example import get_paths

random.seed(123)
np.random.seed(123)

curr_dir = os.getcwd()
config_file = os.path.join(curr_dir, 'configs.ini')
config = configparser.ConfigParser()
config.read(config_file)
API_KEY = config['GENERAL'].get('API_KEY')
UPDATE_INTERVAL = config['GENERAL'].getint('UPDATE_INTERVAL')

# APP PROPERTIES
app = dash.Dash(name='UnWaste! FrontEnd')
app.title = "UnWaste! Project"

# EXTERNAL SETTINGS
# should be loaded elsewhere and imported here / in a submodule

START_COORDS = (41.89117549369146, 12.502362854652286)
# positions are divided in groups for each garbage-truck
df = pd.read_csv('./DATABASE/coords_groups.csv')
POSITIONS = df[['latitude','longitude']].values
GARBAGE_TRUCKS = {0: [(41.89386551419458, 12.501966140302937),
                    (41.89257171020426, 12.501601359887946),
                    (41.8934182762729, 12.501150748787078),
                    (41.892747413518585, 12.50024952658534),
                    (41.892475871829106, 12.501322410158838),
                    (41.892747413518585, 12.502438209075272),
                    (41.89300298111296, 12.501837394274116),
                    (41.89268352146019, 12.5032750582626),
                    (41.89279533252039, 12.50404753443552),
                    (41.89292311634958, 12.504905841294317),
                    (41.89201265098866, 12.504691264579616),
                    (41.89332243916796, 12.50394024607817),
                    (41.89253976409517, 12.498597285882155)
                    ],
                    1: [
                        (41.89067088890724, 12.502566955104093),
                        (41.891597346689515, 12.502116344003225),
                        (41.89177305268408, 12.50003494987064),
                        (41.891405666871506, 12.501107833444138),
                        (41.89067088890724, 12.501300952487366),
                        (41.89174110617557, 12.50409044977846),
                        (41.891293853378684, 12.499391219726544),
                        (41.891261906630504, 12.498726031910977),
                        (41.89011181305501, 12.50421919580728),
                        (41.890638941847506, 12.505485198424006),
                        (41.891868892113756, 12.502481124418214),
                            ]}

GARBAGE_LABELS = []
for k in GARBAGE_TRUCKS.keys():
    GARBAGE_LABELS.append({'label': f'Truck #{k + 1}', 'value': str(k)})
SHOW_ROUTES = {0: False, 1: False}

# precompute paths
paths = get_paths(len(GARBAGE_TRUCKS))

clnt = client.Client(key=API_KEY) # Create client with api key

# theoretically, the paths / markups should be loaded in real time from some service
# in this demo, however, we can just add them from a database of precoumputed positions
@app.callback(Output('map', 'srcDoc'),
              Input('interval-component', 'n_intervals'))  # add an input here to load the pathon the map at a user's notice
def update_map(n):
    global START_COORDS, POSITIONS, GARBAGE_TRUCKS, SHOW_ROUTES, paths

    rome_map = folium.Map(location = START_COORDS, title = "Rome", zoom_start = 16, min_zoom = 16, max_zoom = 18)

    #add garbage bins maps
    for p in POSITIONS:
        folium.Marker(location=[p[0], p[1]],
                    icon = folium.features.CustomIcon("assets\dustbin.png",
                                                    icon_size=(20, 20))
                    ).add_to(rome_map)

    #take trucks position, add it to maps
    for truck in GARBAGE_TRUCKS:

        truck_pos = GARBAGE_TRUCKS[truck][n % len(GARBAGE_TRUCKS[truck])][0], GARBAGE_TRUCKS[truck][n % len(GARBAGE_TRUCKS[truck])][1]

        folium.Marker(location=[truck_pos[0], truck_pos[1]],
                                icon = folium.features.CustomIcon("assets\garbagetruck.png",
                                                                   icon_size=(35, 35)),
                                                                   popup=f'Garbage truck #{truck}'
                     ).add_to(rome_map)

    # get directions
    if False: #SHOW_ROUTES[truck]:  # decomment to use API; TODO: add checkbox to toggle route drawing
        # coordinates = [[truck_pos[1], truck_pos[0]]] + [[POSITIONS[idx][1], POSITIONS[idx][0]] for idx in paths[truck]]
        coordinates = [[POSITIONS[idx][1], POSITIONS[idx][0]] for idx in paths[truck]]
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
data = {"Report_id":[000,111,222,333], 
        "Truck_id":[234,567,876,766],
         "Type":['Report', "Issiue", "NaN", "Injury"],
         "Operator": ['Anil',"Giuliano", "Marco", "Alberto" ]}

df = pd.DataFrame(data)

choose_len = 20

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
                                            html.H1("UnWaste!", className = "header-title"),
                                            html.P('Demo dashboard', className = 'header-description')
                                            ],
                                className = 'header'
                                ),
                #Body1#
                        html.Div(children = [
                                            #Map#
                                            html.Div([
                                                        html.H3('Path Map',  className = 'wintitle'),
                                                        html.Iframe(id = 'map', srcDoc = None, className = 'inframe_map' ),
                                            #Button div
                                                        html.Div([dcc.Dropdown(id='input-on-submit', options = GARBAGE_LABELS, value='0'),
                                                                    html.Button('Submit', id='submit-val', n_clicks=0),
                                                                    html.Div(id='ignore-me', hidden=True)
                                                                 ])
                                                     ],
                                                        
                                                    className = "Map"),
                                        
                                            #Report#
                                            html.Div([
                                                    # reportTitle
                                                    html.H3('Real-Time Reports',  className = 'wintitle'),
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
                                            
                                        className = 'wrapper'),
                        ## body 2
                        html.Div(children = [
                                            ##left-stats
                                            html.Div([ 
                                                    html.H3('Real-time Trucks stats', className = 'wintitle'),
                                                    html.P("Names:"),
                                                    dcc.Dropdown(
                                                                id='names', 
                                                                value='truck_fuel_situation', 
                                                                options=[{'value': x, 'label': x} for x in ['truck_fuel_situation', 'time']],
                                                                clearable=False
                                                                ),
                                                    html.P("Values:"),
                                                    dcc.Dropdown(
                                                                id='values', 
                                                                value='Fuel_L', 
                                                                options=[{'value': x, 'label': x} for x in ['Fuel_L', 'mount_mc']],
                                                                clearable=False
                                                                ),
                                                    
                                                    dcc.Graph(id="pie-chart"),
                                                  
                                                    ],
                                                    className = "left-stats"),
                                            #right Stats
                                            html.Div([
                                                    html.H3('Real-Time bin stats',  className = 'wintitle'),
                                        
                                                    html.P("Waste:"),
                                                    dcc.Dropdown(
                                                                id='waste', 
                                                                value='PET', 
                                                                options=[{'value': x, 'label': x} for x in wastes],
                                                                clearable=False
                                                                ),

                                                       dcc.Graph(id="histo"),
                                                            

                                                    ],
                                                    className="right-stats")
                                            

                            



                                            ], 
                                            className = "wrapper"),
                                  
                        ],
className="HTML"
)





if __name__ == "__main__":
    app.run_server(debug=True)