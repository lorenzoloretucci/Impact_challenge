import dash 
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input
import dash_table

print(dash.__version__)
print(pd.__version__)




#Dataframe 
data = {"Report_id":[000,111,222,333], 
        "Truck_id":[234,567,876,766],
         "Type":['Report', "Issiue", "NaN", "Injury"],
         "Operator": ['Anil',"Giuliano", "Marco", "Alberto" ]}

df = pd.DataFrame(data)

app=dash.Dash(__name__)
app.title = "Prova!"

app.layout = html.Div(
    #MAIN
            children = [    
                #header#
                        html.Div(children = [
                                            html.H1("UnWaste!"),
                                            html.P('Demo dashboard')
                                            ],
                                className = 'Header'
                                ),
                #Body#
                        html.Div(children = [
                                            #Map#
                                            html.Div([
                                                        html.H3('Path Map'),
                                                        html.Iframe(id = 'map', srcDoc = None, height = "500", width = '500'),
                                                        dcc.Input(id='input-on-submit', type='text', value='0'),
                                                        html.Button('Submit', id='submit-val', n_clicks=0),
                                                     ],
                                                        style={"float":"left", 'width':500, 'height': 500, "margin":100},
                                                    className = "Map"),
                                        
                                            #Report#
                                            html.Div([
                                                        # reportTitle
                                                    html.H3('Real-Time Reports'),
                                                     #Table
                                                    dash_table.DataTable(id='table',columns=[{"name": i, "id": i} for i in df.columns],data=df.to_dict('records')),
                                                    
                                                    ],
                                                    style={"float":"left", 'width':500, 'height': 500, "margin": 100}),
                                            html.Div([


                                                    ]),
                                            ],
                                            
                                        className = 'Body',
                                        ),      
                        ]
)
if __name__ == "__main__":
    app.run_server(debug=True)