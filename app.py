import dash 
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import folium
from dash.dependencies import Output, Input
from streamz.dataframe import PeriodicDataFrame

# APP PROPERTIES
app = dash.Dash(name='UnWaste! FrontEnd')
app.title = "UnWaste! Project"

# EXTERNAL SETTINGS
# should be loaded elsewhere and imported here / in a submodule
START_COORDS = (41.89117549369146, 12.502362854652286)
POSITIONS = [(41.8908906679924, 12.502572097053521), 
            (41.891579262511144, 12.502176550683817), 
            (41.8924103430201, 12.503031390817107), 
            (41.893027676367694, 12.501927753222729)]
UPDATE_INTERVAL = 5 * 1000 # milliseconds

# theoretically, the paths / markups should be loaded in real time from some service
# in this demo, however, we can just add them from a database of precoumputed positions
@app.callback(Output('map', 'srcDoc'),
              Input('interval-component', 'n_intervals'))  # add an input here to load the pathon the map at a user's notice
def update_map(n):
    global START_COORDS, POSITIONS

    rome_map = folium.Map(location = START_COORDS, title = "Rome", zoom_start = 16, min_zoom = 16, max_zoom = 16)

    for p in POSITIONS: 
        folium.Marker(location=[p[0], p[1]], icon = folium.features.CustomIcon("assets\dustbin.png",icon_size=(25, 25))).add_to(rome_map)

    if n  % len(POSITIONS) > 0:
        folium.PolyLine(POSITIONS[0: (n % len(POSITIONS) + 1)], color='red', weight=10, opacity=0.8).add_to(rome_map)

    # rome_map.save('map.html')
    # return open("map.html", "r").read()

    return rome_map._repr_html_()

app.layout = html.Div(
    children = [
        html.Div(children = [html.H1("UnWaste!"),html.P('Demo dashboard)', className = 'Header')]),
        html.Div(children = [html.Iframe(id = 'map', srcDoc = None, width = "50%", height = "500", style={'display': 'inline-block'}),
                             dcc.Graph(id="graph2",style={'display': 'inline-block', 'height' : "600"})]),
        dcc.Interval(
            id='interval-component',
            interval=UPDATE_INTERVAL,
            n_intervals=0
        )
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
