import folium 
import pandas as pd


rome_coo = (41.90, 12.49)
position = [(41.8908906679924, 12.502572097053521), 
            (41.891579262511144, 12.502176550683817), 
            (41.8924103430201, 12.503031390817107), 
            (41.893027676367694, 12.501927753222729)]

rome_map = folium.Map(location = rome_coo, title = "Stamen Toner", zoom_start = 12)

for p in position: 
    folium.Marker(location=[p[0], p[1]], 
    icon = folium.features.CustomIcon("assets\dustbin.png",icon_size=(25, 25)) ).add_to(rome_map)

folium.PolyLine(position[0:2],
                color='red',
                weight=15,
                opacity=0.8).add_to(rome_map)



rome_map.save('ROme.html')