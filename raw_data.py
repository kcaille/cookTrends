import streamlit as st
import pandas as pd

import time

def app():

##### Loading data ##################################################################
	#@st.cache
	def load_recipes():
		# recipes
		fn = './recipes_noel_+_page1to9.csv'
		recipes_df = pd.read_csv(fn)
		recipes_df.drop(columns = 'Unnamed: 0', inplace = True)
		return recipes_df
	
	#@st.cache
	def load_comments():
		# comments
		fn = './comments_noel_+_page1to9.csv'
		comments_df = pd.read_csv(fn)
		comments_df.drop(columns = 'Unnamed: 0', inplace = True)
		return comments_df

	# Create a text element and let the reader know the data is loading.
	with st.spinner('Chargement des données...'):
	
		# Load data into the dataframe.
		comments_df = load_comments()
		recipes_df = load_recipes()

	# Notify the reader that the data was successfully loaded.
	st.success('Données chargées!')
####################################################################################
	
	st.subheader('Données source')

	with st.beta_expander("Voir les données sur les recettes"):
		st.dataframe(recipes_df,height = 500)

	with st.beta_expander("Voir les données sur les commentaires"):
		st.write(comments_df, height = 500)
