import pandas as pd
import numpy as np
import scipy as sp
import random

import datetime as dt
from datetime import datetime


from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import re
import string
import spacy

import streamlit as st

import nltk
nltk.download('punkt')
nltk.download('stopwords')


# On importe quelques librairies spécifiques
from PIL import Image
from wordcloud import WordCloud
from palettable.colorbrewer.sequential import Blues_9




#TITLE
st.image('./CookTrends logo.png')
st.title('cook.Trends')
st.write('')


#Loading dataset
#@st.cache
def load_comments():
	# comments
	fn = './comments_noel_+_page1to9.csv'
	comments_df = pd.read_csv(fn)
	comments_df.drop(columns = 'Unnamed: 0', inplace = True)
	comments_df.comments_date = pd.to_datetime(comments_df.comments_date)
	#comments_df.comments_date = comments_df.comments_date.tz_convert('Europe/Berlin')
	return comments_df

#@st.cache
def load_recipes():
	# recipes
	fn = './recipes_noel_+_page1to9.csv'
	recipes_df = pd.read_csv(fn)
	recipes_df.drop(columns = 'Unnamed: 0', inplace = True)
	return recipes_df

def load_commentsNoNan():
	# recipes
	fn = './commentsV3_sans_nanV3.csv'
	comments_nonan_df = pd.read_csv(fn)
	return comments_nonan_df

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Chargement des données...')

# Load 10,000 rows of data into the dataframe.
comments_df = load_comments()
recipes_df = load_recipes()
comments_nonan_df = load_commentsNoNan()

# Notify the reader that the data was successfully loaded.
data_load_state.text("Données chargées!")


#MAIN

#SIDEBAR
st.sidebar.subheader('Choisir la période pour les tendances')

start_day = st.sidebar.slider('Start day ?', min_value = datetime(2020,1,1, 0, 00), max_value = datetime(2021,1,11, 0, 00), value = datetime(2020,1,1, 0, 00), format = "DD/MM/YYYY")
st.sidebar.write("Start day:", start_day)

end_day = st.sidebar.slider('End day ?', 
	min_value = datetime(2020,1,1, 0, 00), max_value = datetime(2021,1,11, 0, 00), value = datetime(2021,1,1, 0, 00), 
	format = "DD/MM/YYYY")

st.sidebar.write("End day:", end_day)



#RAW DATA
if st.checkbox('Voir les données source'):
	st.subheader('Données source')
	#st.write(comments_df)
	st.write(recipes_df)

st.write('-' * 100)

###############################################################################

st.header('Activité par mois')
if st.checkbox("Voir l'activité par mois"):

	#ACTIVITY
	st.subheader('Activité')


	st.subheader('Nombre de commentaires par mois - 2020')

	def getActivity():
	  	comments_df['year'] = comments_df['comments_date'].dt.year
	  	comments_df['month'] = comments_df['comments_date'].dt.month

	  	comments_grouped= comments_df.groupby(['year', 'month'], as_index= False).count().iloc[-25:-1]

	  	return comments_grouped[['year','month', 'recipe_url']]

	chart_data = getActivity()

	#fig, ax = plt.subplots(figsize = (12,8))

	#ax.plot(chart_data["year_month"], chart_data["recipe_url"])


	# Create an array with the colors you want to use
	colors = ["#FAB57C", "#D09170"]

	# Set your custom color palette
	customPalette = sns.color_palette(colors)

	fig, ax = plt.subplots(figsize = (12,8))

	g = sns.lineplot(data=chart_data[chart_data['year'] == 2020], x="month", y="recipe_url", hue = "year", ax = ax, color = "#FAB57C")
	#plot(chart_data["year_month"], chart_data["recipe_url"])

	ax.set_xlabel('Mois', fontsize=17, color='dimgrey')
	ax.set_ylabel('Nombre de commentaires',  fontsize=17, color='dimgrey')
	plt.xticks(np.arange(1,13), ['Jan', 'Fev', 'Mars','Avril', 'Mai', 'Juin', 'Juill', 'Aout', 'Sept', 'Oct', 'Nov', 'Dec'])
	ax.set_yticks(np.arange(0, 5001, 1000), ('1 000', '2 000', '3 000', '4 000', '5 000'))

	for s in ["top","right"]:
	    ax.spines[s].set_visible(False)

	st.pyplot(fig)

	st.subheader('Moyenne des like par mois - 2020')

	def getMeanRating():
	  comments_df['year'] = comments_df['comments_date'].dt.year
	  comments_df['month'] = comments_df['comments_date'].dt.month

	  comments_grouped= comments_df.groupby(['year', 'month'], as_index= False).mean().iloc[-25:-1]

	  return comments_grouped[['year','month', 'comments_rating']]

	chart_data = getMeanRating()

	#fig, ax = plt.subplots(figsize = (12,8))

	#ax.plot(chart_data["year_month"], chart_data["recipe_url"])


	# Create an array with the colors you want to use
	colors = ["#FAB57C", "#D09170"]

	# Set your custom color palette
	customPalette = sns.color_palette(colors)

	fig, ax = plt.subplots(figsize = (12,8))

	g = sns.barplot(data=chart_data[chart_data['year'] == 2020], x="month", y="comments_rating", ax = ax, color = "#FAB57C")
	#plot(chart_data["year_month"], chart_data["recipe_url"])

	ax.set_xlabel('Mois', fontsize=17, color='dimgrey')
	ax.set_ylabel('Nombre de commentaires',  fontsize=17, color='dimgrey')
	ax.set_xticks(np.arange(1,13), ('Jan', 'Fev', 'Mars','Avril', 'Mai', 'Juin', 'Juill', 'Aout', 'Sept', 'Oct', 'Nov', 'Dec'))



	for s in ["top","right"]:
	    ax.spines[s].set_visible(False)

	st.pyplot(fig)

###############################################################################

st.write('-' * 100)

###############################################################################
st.header('Tendances')
st.write('Choisir à gauche la période sur laquelle vous voulez mesurer les tendances.')
if st.checkbox("Voir les tendances sur la période choisie"):

	###############################################################################
	#TOP RECETTES
	st.subheader('Top 10 des recettes')

	def dix_meilleures_recettes(date_debut, date_fin):
	  Q=3.5
	  mask = (comments_df['comments_date'] > date_debut) & (comments_df['comments_date'] <= date_fin)
	  new_df = comments_df.loc[mask]
	  df_coms_grouped = new_df.groupby(['recipe_title'], as_index=False).agg({'comments_rating': ['mean', 'count']})
	  df_coms_grouped.columns = list(map('_'.join, df_coms_grouped.columns.values))

	  df_coms_grouped['combined_rating'] = df_coms_grouped['comments_rating_mean'] + 5*(1-np.exp(-df_coms_grouped['comments_rating_count']/Q))
	  
	  return df_coms_grouped.sort_values('combined_rating', ascending=False)[0:10]

	top_10_recipes = dix_meilleures_recettes(start_day, end_day)

	for recipe in top_10_recipes.recipe_title_.values.tolist():
		st.write(recipe)

	st.write('-' * 100)

	###############################################################################
	# Ce qui fait une bonne recette
	st.subheader('Ce qui rend les recettes tendances')

	# tri décroissant du poids
	df_coms_ratings_decroissant2 = comments_nonan_df.sort_values(by='comments_rating', ascending=False)

	liste_elements = []
	for element in df_coms_ratings_decroissant2['comments_content'][0:30]:
	  # print(element)
	  liste_elements.append(element)
	  texte_com2 = " ".join(liste_elements)

	# en tokens de mots
	texte_com_tok = nltk.word_tokenize(texte_com2)


	texte_com_tok_freqdist = nltk.FreqDist(texte_com_tok) # fréquence d'apparition de chaque mot


	top_30 = texte_com_tok_freqdist.most_common(20) # les 20 mots les plus communs

	df_top30 = pd.DataFrame(top_30)
	df_top30.columns = ['Mots', 'Fréquence']


	# Plot barplot

	fig3, ax = plt.subplots(figsize = (12,8))

	g = sns.barplot(x = df_top30.Mots, y = df_top30["Fréquence"], palette="YlGnBu_r")
	#plot(chart_data["year_month"], chart_data["recipe_url"])


	plt.xlabel('\nMots', fontsize=17, color='dimgrey')
	plt.ylabel("Fréquence\n", fontsize=17, color='dimgrey')
	#plt.title("Fréquence des 20 mots les plus utilisés\n", fontsize=20, color='#e74c3c')
	plt.xticks(rotation= 75, fontsize=16)
	#ax.set_xlabel('Mois')
	#ax.set_ylabel('Nombre de commentaires')
	#ax.set_xticks(np.arange(1,13), ('Jan', 'Fev', 'Mars','Avril', 'Mai', 'Juin', 'Juill', 'Aout', 'Sept', 'Oct', 'Nov', 'Dec'))



	for s in ["top","right"]:
	    ax.spines[s].set_visible(False)

	st.pyplot(fig3)

	###############################################################################
	#TOP INGREDIENTS
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

	st.write('-' * 100)





	###############################################################################
	#FIND INGREDIENTS TREND
	st.subheader('Trouver les recettes correspondants à un ingrédients')

	veg = st.text_input('Name')
	if not veg:
		st.warning('Please input a name.')
		st.stop()
		st.success('Thank you for inputting a name.')

	for row in range(len(data_merged[data_merged['ingredient_main'].str.contains(veg)])):
		try:
			st.write(data_merged[data_merged['ingredient_main'].str.contains(veg)].iloc[row, 1])
			st.image(data_merged[data_merged['ingredient_main'].str.contains(veg)].iloc[row, 8])
		except:
			pass


