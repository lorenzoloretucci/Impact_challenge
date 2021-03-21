import dash 
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input

print(dash.__version__)

data = pd.read_csv('avocado.csv')

data["Date"] = pd.to_datetime(data["Date"], format ="%Y-%m-%d" )
data.sort_values("Date", inplace = True)

app = dash.Dash(__name__)
app.title = "UnWaste! Project"
