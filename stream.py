import streamlit as st
from streamlit_folium import folium_static
import folium
import numpy as np 

data = np.random.randn(10, 1)


st.set_page_config(layout = 'wide')

st.title('UnWaste!')
c1, c2 = st.beta_columns((1, 1))

c1.title('Map')

m = folium.Map(location = [39.949610, -75.150282], zoom_start = 16)

tooltip = 'Liberty Bell'
folium.Marker(
    [39.949610, -75.150282], popup = 'Liberty Bell', tooltip = tooltip
).add_to(m)

with c1:
    folium_static(m)

with c2:
    c2.title('Report')
    c2.write(data)