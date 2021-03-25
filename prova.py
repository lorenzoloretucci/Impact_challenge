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
POSITIONS = {0: [(41.8908906679924, 12.502572097053521), 
            (41.891579262511144, 12.502176550683817), 
            (41.8924103430201, 12.503031390817107), 
            (41.893027676367694, 12.501927753222729)],
            1: [
                (42.8924103430201, 12.503031390817107)
            ]}
GARBAGE_TRUCKS = {0: [(41.89094733558171, 12.505344489064024),
                      (41.890840034189985, 12.504987924351767),
                      (41.890631075681256, 12.504282384995024),
                      (41.89046164764188, 12.503819614718148)],
                    1: [(41.890631075681256, 12.504282384995024),
                        (41.89094733558171, 12.505344489064024),
                        (41.89046164764188, 12.503819614718148),
                        (41.890840034189985, 12.504987924351767)]}

GARBAGE_LABELS = []
for k in GARBAGE_TRUCKS.keys():
    GARBAGE_LABELS.append({'label': f'Truck #{k + 1}', 'value': str(k)})
SHOW_ROUTES = {0: False, 1: False}


clnt = client.Client(key=API_KEY) # Create client with api key

# theoretically, the paths / markups should be loaded in real time from some service
# in this demo, however, we can just add them from a database of precoumputed positions
@app.callback(Output('map', 'srcDoc'),
              Input('interval-component', 'n_intervals'))  # add an input here to load the pathon the map at a user's notice
def update_map(n):
    global START_COORDS, POSITIONS, GARBAGE_TRUCKS, SHOW_ROUTES

    rome_map = folium.Map(location = START_COORDS, title = "Rome", zoom_start = 16, min_zoom = 16, max_zoom = 16)

    #take trucks position, add it to maps
    for truck in GARBAGE_TRUCKS:
        truck_pos = GARBAGE_TRUCKS[truck][n % len(GARBAGE_TRUCKS[truck])][0], GARBAGE_TRUCKS[truck][n % len(GARBAGE_TRUCKS[truck])][1]

        folium.Marker(location=[truck_pos[0], truck_pos[1]],
                                icon = folium.features.CustomIcon("assets\garbagetruck.png",
                                                                   icon_size=(25, 25)),
                                                                   popup=f'Garbage truck #{truck}'
                     ).add_to(rome_map)
        #add garbage bins maps
        for p in POSITIONS[truck]: 
            folium.Marker(location=[p[0], p[1]],
                          icon = folium.features.CustomIcon("assets\dustbin.png",
                                                            icon_size=(30, 30))
                          ).add_to(rome_map)

        # get directions
        if False: #SHOW_ROUTES[truck]:  # decomment to use API; TODO: add checkbox to toggle route drawing
            coordinates = [[truck_pos[1], truck_pos[0]]] + [[p[1], p[0]] for p in POSITIONS[truck]]
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
    fig = px.pie(df_left, values=values, names=names)
    return fig

##### callback for right- stats ###
@app.callback(
    Output("graph", "figure"), 
    [Input("mean", "value"), 
     Input("std", "value")])
def display_color(mean, std):
    data = np.random.normal(mean, std, size=500)
    fig = px.histogram(data, nbins=30, range_x=[-10, 10])
    return fig





###############  Dataframe ############################## 
data = {"Report_id":[000,111,222,333], 
        "Truck_id":[234,567,876,766],
         "Type":['Report', "Issiue", "NaN", "Injury"],
         "Operator": ['Anil',"Giuliano", "Marco", "Alberto" ]}

df = pd.DataFrame(data)

np.random.seed(2020)

data_left = {'Fuel':[]}

df_left = px.data.tips()
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
                                                        html.H3('Path Map'),
                                                        html.Iframe(id = 'map', srcDoc = None, height = "305", width = '500', className = 'inframe_map' ),
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
                                                    html.H3('Real-Time Reports'),
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
                                                    html.H3('Real-time Trucks stats'),
                                                    html.P("Names:"),
                                                    dcc.Dropdown(
                                                                id='names', 
                                                                value='day', 
                                                                options=[{'value': x, 'label': x} for x in ['smoker', 'day', 'time', 'sex']],
                                                                clearable=False
                                                                ),
                                                    html.P("Values:"),
                                                    dcc.Dropdown(
                                                                id='values', 
                                                                value='total_bill', 
                                                                options=[{'value': x, 'label': x} for x in ['total_bill', 'tip', 'size']],
                                                                clearable=False
                                                                ),
                                                    
                                                    dcc.Graph(id="pie-chart"),
                                                  
                                                    ],
                                                    className = "left-stats"),
                                            #right Stats
                                            html.Div([
                                                    html.H3('Real-Time bin stats'),
                                                       dcc.Graph(id="graph"),
                                                            html.P("Mean:"),
                                                            dcc.Slider(id="mean", min=-3, max=3, value=0, 
                                                                    marks={-3: '-3', 3: '3'}),
                                                            html.P("Standard Deviation:"),
                                                            dcc.Slider(id="std", min=1, max=3, value=1, 
                                                                    marks={1: '1', 3: '3'}),

                                                    ],
                                                    className="right-stats")
                                            

                            



                                            ], 
                                            className = "wrapper"),
                                  
                        ],
className="HTML"
)





if __name__ == "__main__":
    app.run_server(debug=True)