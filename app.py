import dash 
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import folium 
from dash.dependencies import Output, Input
import schedule 
import threading, time 

print(dash.__version__)




app = dash.Dash(__name__)
app.title = "UnWaste! Project"

app.layout = html.Div(
    children = [
        html.Div(children = [html.H1("UnWaste!"),html.P('Prova della dashboard!')], className = 'Header'),
        html.Div(children = [html.Iframe(id = 'map', srcDoc = open("ROme.html", "r").read(), width = "50%", height = "500", style={'display': 'inline-block'}),
                             dcc.Graph(id="graph2",style={'display': 'inline-block', 'height' : "600"})])    
    ]
)


#@app.callback(dash.dependencies.Output('map', 'srcDoc'))
def update_map():
    print("Dai che vaaaaa")
#    return open('ROme.html', 'r').read()

schedule.every(10).seconds.do(update_map)

def my_thread_proc():
    while True:
        schedule.run_pending()
        time.sleep(1)

t = threading.Thread(target = my_thread_proc, daemon = True)

if __name__ == "__main__":
    app.run_server(debug=True)
    t.run()

