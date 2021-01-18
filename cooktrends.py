pip install scipy

import streamlit as st

# On import les 'apps'
import raw_data
import general_activity
import cooking_trend
import index

import pandas as pd

import datetime as dt
from datetime import datetime



# Source: https://medium.com/@u.praneel.nihar/building-multi-page-web-app-using-streamlit-7a40d55fa5b4
# SIDEBAR
#Variables
PAGES = {
    "Accueil": index,
    "Données Sources": raw_data,
    "Analyseur d'activité": general_activity,
    "Analyseur de tendance": cooking_trend
}

#Header
st.sidebar.title('cook.Trends')
st.sidebar.image('./CookTrends logo.png')


#Navigation
st.sidebar.title('Navigation')
selection = st.sidebar.radio("", list(PAGES.keys()))


#Credits
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.title('A  propos')
st.sidebar.info("Cette app a été créée pour le datathon organisé par la Wild Code School en partenariat avec Search Foresight.   \n Elle a été réalisée par :   \nAlexandrine NGANKIMA, Olivier ROHR, Valentin PASQUIER, Boris LACQUEMAND, Kevin CAILLE.")


#App
page = PAGES[selection]
page.app()