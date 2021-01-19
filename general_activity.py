import streamlit as st

import pandas as pd
import numpy as np
import scipy as sp
import random

import matplotlib.pyplot as plt
import seaborn as sns


from PIL import Image
from wordcloud import WordCloud
from palettable.colorbrewer.sequential import Blues_9



def app():

##### Loading data ##################################################################
	#@st.cache
	def load_comments():
		# comments
		fn = './data/comments_noel_+_page1to9.csv'
		comments_df = pd.read_csv(fn)
		comments_df.drop(columns = 'Unnamed: 0', inplace = True)
		#comments_df.comments_date = comments_df.comments_date.tz_convert('Europe/Berlin')
		return comments_df

	# Create a text element and let the reader know the data is loading.
	with st.spinner('Chargement des données...'):
	
		# Load data into the dataframe.
		comments_df = load_comments()


	# Notify the reader that the data was successfully loaded.
	st.success('Données chargées!')
####################################################################################


##### Comments #####################################################################
	st.header('Activité par mois')
	st.subheader('Nombre de commentaires par mois - 2020')

	def getActivity():
		comments_df_wf = comments_df
		comments_df_wf.comments_date = pd.to_datetime(comments_df_wf.comments_date)
		comments_df_wf['year'] = comments_df_wf['comments_date'].dt.year
		comments_df_wf['month'] = comments_df_wf['comments_date'].dt.month

		comments_grouped= comments_df_wf.groupby(['year', 'month'], as_index= False).count().iloc[-25:-1]

		return comments_grouped[['year','month', 'recipe_url']]

	chart_data = getActivity()


	# Create an array with the colors you want to use
	colors = ["#FAB57C", "#D09170"]

	# Set your custom color palette
	customPalette = sns.color_palette(colors)

	#Creating the figure
	fig, ax = plt.subplots(figsize = (8, 6))

	g = sns.lineplot(data=chart_data[chart_data['year'] == 2020], x="month", y="recipe_url", ax = ax, color = "orange", linewidth = 5)

	#ax.set_xlabel('Mois', fontsize=17, color='dimgrey', labelpad=15)
	ax.set_xlabel('')
	ax.set_ylabel('')

	ax.set_xticks(np.arange(1,13))
	ax.set_xticklabels(['Jan', 'Fev', 'Mars','Avril', 'Mai', 'Juin', 'Juill', 'Aout', 'Sept', 'Oct', 'Nov', 'Dec'])
	ax.set_yticks(np.arange(0, 5001, 1000))
	ax.set_yticklabels(['0', '1 000', '2 000', '3 000', '4 000', '5 000'])

	ax.tick_params(axis='x', colors = 'dimgrey')
	ax.tick_params(axis = 'y', colors = 'grey')

	ax.yaxis.grid(True)

	for s in ["top","right", "left"]:
	    ax.spines[s].set_visible(False)

	ax.spines['bottom'].set_color('dimgrey') 

	st.write(fig)

	with st.beta_expander("Plus d'infos"):
		st.write('Ce graphique représente le nombre de commentaires postés chaque mois sur le site Marmiton.')

####################################################################################
	st.write('-' * 100)
####################################################################################

##### Likes ########################################################################
	st.subheader('Moyenne des Likes par mois - 2020')

	def getMeanRating():
		comments_df_wf = comments_df
		comments_df_wf.comments_date = pd.to_datetime(comments_df_wf.comments_date)
		comments_df_wf['year'] = comments_df['comments_date'].dt.year
		comments_df_wf['month'] = comments_df['comments_date'].dt.month
		
		comments_grouped= comments_df_wf.groupby(['year', 'month'], as_index= False).mean().iloc[-25:-1]

		return comments_grouped[['year','month', 'comments_rating']]

	chart_data = getMeanRating()

	# Create an array with the colors you want to use
	colors = ["#FAB57C", "#D09170"]

	# Set your custom color palette
	customPalette = sns.color_palette(colors)

	fig, ax = plt.subplots(figsize = (8,6))

	g = sns.barplot(data=chart_data[chart_data['year'] == 2020], x="month", y="comments_rating", ax = ax, color = "cornflowerblue")
	#plot(chart_data["year_month"], chart_data["recipe_url"])

	ax.set_xlabel('')
	ax.set_ylabel('')

	#ax.set_xticks(np.arange(1,13))
	ax.set_xticklabels(['Jan', 'Fev', 'Mars','Avril', 'Mai', 'Juin', 'Juill', 'Aout', 'Sept', 'Oct', 'Nov', 'Dec'])
	ax.set_yticks(np.arange(0, 6, 1))
	ax.set_yticklabels(['0', '1', '2', '3', '4', '5'])

	ax.tick_params(axis='x', colors = 'dimgrey')
	ax.tick_params(axis = 'y', colors = 'grey')

	ax.yaxis.grid(True)

	for s in ["top","right", "left"]:
	    ax.spines[s].set_visible(False)

	ax.spines['bottom'].set_color('dimgrey') 

	st.write(fig)

	with st.beta_expander("Plus d'infos"):
		st.write('Ce graphique représente les notes moyennes des commentaires postés chaque mois sur le site Marmiton.')
####################################################################################