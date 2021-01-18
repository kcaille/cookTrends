import streamlit as st

def app():
	st.image('./CookTrends logoLarge.png')
	st.write('-' * 80)
	st.title('Bienvenue sur cookTrends')
	st.write("cookTrends vous permet d'analyser les tendances en termes de cuisine en se basant sur les commentaires postés sur le site marmiton. Les recettes tendances sont calculées grâce à notre cookTrend score qui intègre à la fois le nombre de commentaire et les notes de ces commentaires.")
	st.write("Par exemple:")
	st.write("- Quelles sont les recettes les plus tendances sur la périodes de noël ?")
	st.write("- Quels ingrédients retrouve-t-on dans ces recettes ?")
