pip install scipy

import streamlit as st

import pandas as pd
import numpy as np
import scipy as sp
import random

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from wordcloud import WordCloud
from palettable.colorbrewer.sequential import Blues_9

# NLP
import re
import string
import spacy

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


import datetime as dt
from datetime import datetime

import webbrowser

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
		comments_df.comments_date = pd.to_datetime(comments_df.comments_date)
		return comments_df

	#@st.cache
	def load_commentsNoNan():
		# recipes
		fn = './commentsV3_sans_nanV3.csv'
		comments_nonan_df = pd.read_csv(fn)
		comments_nonan_df.comments_date = pd.to_datetime(comments_nonan_df.comments_date)
		return comments_nonan_df

	# Create a text element and let the reader know the data is loading.
	with st.spinner('Chargement des données...'):
	
		# Load data into the dataframe.
		comments_df = load_comments()
		recipes_df = load_recipes()
		comments_nonan_df = load_commentsNoNan()

	# Notify the reader that the data was successfully loaded.
	st.success('Données chargées!')
####################################################################################
	


##### Time selection  ##################################################################
	st.header('Tendances')
	st.write('Vous pouvez ici analyser les recettes et les ingrédients les plus tendances sur une période donnée:')

	#SIDEBAR
	st.subheader('Choisir la période souhaitée')
	start_day = st.date_input('Date de début', value = datetime(2020,1,1), min_value = datetime(2020,1,1), max_value = datetime(2021,1,10))
	end_day = st.date_input('Date de fin', value = datetime(2021,1,10), min_value = datetime(2020,1,1), max_value = datetime(2021,1,10))
	'''
	start_day = st.slider('Start day ?', min_value = datetime(2020,1,1, 0, 00), max_value = datetime(2021,1,11, 0, 00), value = datetime(2020,1,1, 0, 00), format = "DD/MM/YYYY")
	st.write("Start day:", start_day)


	end_day = st.slider('End day ?', min_value = datetime(2020,1,1, 0, 00), max_value = datetime(2021,1,11, 0, 00), value = datetime(2021,1,1, 0, 00), 
	format = "DD/MM/YYYY")

	st.write("End day:", end_day)
	'''
	start_day = pd.to_datetime(start_day)
	end_day = pd.to_datetime(end_day)


	if st.button('Voir les tendances'):
####################################################################################
		
		st.write('-' * 100)
		st.title('Resultats')

####TOP RECETTES###########################################################################
		with st.beta_expander("Top 10 des recettes les plus tendances sur la période"):

			def dix_meilleures_recettes(date_debut, date_fin):
				Q= 100
				mask = (comments_df['comments_date'] > start_day) & (comments_df['comments_date'] <= end_day)
				new_df = comments_df.loc[mask]
				df_coms_grouped = new_df.groupby(['recipe_url', 'recipe_title'], as_index=False).agg({'comments_rating': ['mean', 'count']})
				df_coms_grouped.columns = list(map('_'.join, df_coms_grouped.columns.values))

				df_coms_grouped['combined_rating'] = df_coms_grouped['comments_rating_mean'] + 5*(1-np.exp(-df_coms_grouped['comments_rating_count']/Q))
				df_coms_grouped['combined_rating'] = round(df_coms_grouped['combined_rating'], 1)

				return df_coms_grouped.sort_values('combined_rating', ascending=False)[0:10]


			top_10_recipes = dix_meilleures_recettes(start_day, end_day)

			for row in range(10):
				cooktrend_score = f"cookTrendscore:  {top_10_recipes.iloc[row, 4]} / 10"
				st.subheader(top_10_recipes.iloc[row, 1])
				st.write(cooktrend_score)
				st.write(top_10_recipes.iloc[row, 0])
############################################################################################		

		st.write('-' * 100)

####TOP RECETTES - Zoom MOTS#############################################################
		with st.beta_expander("Quels mots retrouve-t-on dans les commentaires des recettes tendances ?"):

			def trente_meilleures_recettes(date_debut, date_fin):
				Q= 100
				mask = (comments_df['comments_date'] > start_day) & (comments_df['comments_date'] <= end_day)
				new_df = comments_df.loc[mask]
				df_coms_grouped = new_df.groupby(['recipe_url', 'recipe_title'], as_index=False).agg({'comments_rating': ['mean', 'count']})
				df_coms_grouped.columns = list(map('_'.join, df_coms_grouped.columns.values))

				df_coms_grouped['combined_rating'] = df_coms_grouped['comments_rating_mean'] + 5*(1-np.exp(-df_coms_grouped['comments_rating_count']/Q))
				df_coms_grouped['combined_rating'] = round(df_coms_grouped['combined_rating'], 1)

				return df_coms_grouped.sort_values('combined_rating', ascending=False)[0:30]


			top_30_recipes = trente_meilleures_recettes(start_day, end_day)
			top_30_recipes_list = top_30_recipes['recipe_title_'].values.tolist()


			# tri décroissant du poids
			df_coms_ratings_decroissant = comments_nonan_df.sort_values(by='comments_rating', ascending=False)
			df_coms_ratings_decroissant.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1'], inplace= True)
			mask = (df_coms_ratings_decroissant['comments_date'] >= start_day) & (df_coms_ratings_decroissant['comments_date'] <= end_day)
			df_coms_ratings_decroissant = df_coms_ratings_decroissant[mask]
			df_coms_ratings_decroissant.drop_duplicates(subset = ['comments_content'], keep = 'first', inplace = True)
			

			liste_elements = []

			for element in df_coms_ratings_decroissant[df_coms_ratings_decroissant['recipe_title'].isin(top_30_recipes_list)]['comments_content']:
			  # print(element)
			  liste_elements.append(element)
			  texte_com = " ".join(liste_elements)

			# en tokens de mots
			stop_words = set(stopwords.words('french'))
			personal_stop_words = ['faire', 'mettre', 'rajouter', 'ajouter']
			texte_com_tok = nltk.word_tokenize(texte_com)
			texte_com_tok = [word for word in texte_com_tok if word not in stop_words if word not in personal_stop_words]


			texte_com_tok_freqdist = nltk.FreqDist(texte_com_tok) # fréquence d'apparition de chaque mot


			top_words = texte_com_tok_freqdist.most_common(10) # les 20 mots les plus communs

			top_words = pd.DataFrame(top_words)
			top_words.columns = ['Mots', 'Fréquence']


			# Plot barplot

			fig3, ax = plt.subplots(figsize = (8,6))

			g = sns.barplot(x = top_words.Mots, y = top_words["Fréquence"], palette="YlOrBr_r")
			#plot(chart_data["year_month"], chart_data["recipe_url"])


			#ax.set_xlabel('Mois', fontsize=17, color='dimgrey', labelpad=15)
			ax.set_xlabel('')
			ax.set_ylabel('')

			#ax.set_xticks(np.arange(1,13))
			#ax.set_xticklabels(['Jan', 'Fev', 'Mars','Avril', 'Mai', 'Juin', 'Juill', 'Aout', 'Sept', 'Oct', 'Nov', 'Dec'])
			#ax.set_yticks(np.arange(0, 20, 5))
			#ax.set_yticklabels(['0', '1 000', '2 000', '3 000', '4 000', '5 000'])

			ax.tick_params(axis='x', colors = 'dimgrey')
			ax.tick_params(axis = 'y', colors = 'grey')

			ax.yaxis.grid(True)

			for s in ["top","right", "left"]:
			    ax.spines[s].set_visible(False)

			ax.spines['bottom'].set_color('dimgrey') 

			st.write(fig3)
############################################################################################		

		st.write('-' * 100)

####TOP INGREDIENTS ########################################################################
		with st.beta_expander("Quels ingrédients dans les recettes les plus tendances ?"):
			# On va partir du dataFrame des commentaires que l'on va filtrer
			def filterDF(df, start_date, end_date):
			  df_filtered = df[(df['comments_date'] >= start_date) & (df['comments_date'] <= end_date)]

			  return df_filtered

			# On va ensuite calculer les moyennes, count et note combinée pour chacune des recettes
			def calculateCombinedRating(df):
			  df_grouped = df.groupby('recipe_url', as_index=False).agg({'comments_rating': ['mean', 'count']})
			  df_grouped.columns = list(map('_'.join, df_grouped.columns.values))

			  Q = 3.5
			  df_grouped['combined_rating'] = df_grouped['comments_rating_mean'] + 5*(1-np.exp(-df_grouped['comments_rating_count']/Q))

			  return df_grouped

			# On doit ensuite merger avec le dataset des recettes
			def getMainIngredients(ingredients):
			  ingredients = eval(ingredients)
			  main_ingredients = []

			  for ingredient in ingredients:
			    main_ingredients.append(ingredient[2].lower())

			  return main_ingredients

			#Main
			data_filtered = filterDF(comments_df, start_day, end_day)

			data_grouped = calculateCombinedRating(data_filtered)

			data_merged = pd.merge(left = recipes_df, right = data_grouped, how = 'left', left_on = 'recipe_url', right_on = 'recipe_url_')

			data_merged.dropna(subset=['combined_rating'], inplace = True)

			data_merged['ingredient_main'] = data_merged['ingredients'].apply(lambda x : " ".join(getMainIngredients(x)))


			# On vectorize notre colonne content
			vectorizer = CountVectorizer()
			content_vectorized = vectorizer.fit_transform(data_merged['ingredient_main'])

			# Si un mot apparait plus d'une fois dans un post, on peut avoir une valeur supérieure à 1 hors on veut juste savoir s'il apparait au moins une fois
			content_vectorized_withOnes = (content_vectorized > 0).multiply(1)

			# Pour chaque colonne on multiplie la valeur (1 ou 0) l'engagement score de la ligne
			content_scored_vectorized = sp.sparse.csr_matrix(content_vectorized_withOnes).multiply(sp.sparse.csr_matrix(data_merged['combined_rating'].values).T)

			# On fait la moyenne pour chacune des colonnes - note : la moyenne prend en compte les 0, pas terrible!
			# mean_engagement_score_matrix = content_scored_vectorized.mean(axis = 0)

			# On fait la sum des score d'engagement
			sum_engagement_score_matrix = content_scored_vectorized.sum(axis = 0)

			# On count le nombre de valeurs non zero - note: on le fait dans le content_vectorized afin de bien prendre en compte les mots pour lesquels on n'a pas de score d'engagement
			count_engagement_score_matrix = (content_vectorized_withOnes > 0).sum(axis = 0)

			# On peut calculer la moyenne
			mean_engagement_score_matrix = sum_engagement_score_matrix / count_engagement_score_matrix

			# On convertit notre matrice en array
			mean_engagement_score_array = np.asarray(mean_engagement_score_matrix)[0]

			# On va calculer le nombre d'occurence de chaque mot car un mot qui apparait qu'une fois dans un poste avec énormément d'engagement va avoir un fort impact
			nb_occurence_matrix = content_vectorized.sum(axis= 0)
			nb_occurence_array = np.asarray(nb_occurence_matrix)[0]

			# On crée un dataframe avec les valeurs souhaitées
			df_scores_ingredients = pd.DataFrame({'words': vectorizer.get_feature_names(), 'mean_combined_score': mean_engagement_score_array, 'occurence': nb_occurence_array})


			st.subheader('Top ingrédients')

			df_scores_ingredients['mean_combined_score'] = df_scores_ingredients['mean_combined_score'].astype('int')
			data = dict(zip(df_scores_ingredients['words'].tolist(), df_scores_ingredients['mean_combined_score'].tolist()))

			def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
			    return tuple(Blues_9.colors[random.randint(2,8)])

			icon_path = './dinner.png'
			icon = Image.open(icon_path)
			mask = Image.new("RGB", icon.size, (255,255,255))
			mask.paste(icon,icon)
			mask = np.array(mask)

			wc = WordCloud(background_color="white", max_words=100, mask=mask,
			               max_font_size=300, random_state=42)

			# generate word cloud
			wc.generate_from_frequencies(data)
			wc.recolor(color_func=color_func, random_state=3)
			#wc.to_file("my_wordcloud.png")

			fig2, ax2 = plt.subplots(figsize=(15,8))
			ax2.imshow(wc, interpolation="bilinear")
			ax2.axis("off")
			ax2.margins(x=0, y=0)

			st.pyplot(fig2)
############################################################################################		

		st.write('-' * 100)

####FIND INGREDIENTS TREND ########################################################################
		with st.beta_expander('Trouver les recettes tendances par ingrédients'):

			veg = st.text_input('Name')
			if st.button('Chercher'):
				for row in range(len(data_merged[data_merged['ingredient_main'].str.contains(veg)])):
					st.write(data_merged[data_merged['ingredient_main'].str.contains(veg)].iloc[row, 1])
