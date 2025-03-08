import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- Création d'un jeu de données fictif ---
# Pour l'exemple, on suppose que le prix d'une maison dépend de sa surface (en m²) et du nombre de chambres.
data = {
    "Surface": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    "Chambres": [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    "Prix": [100, 120, 150, 180, 210, 240, 270, 300, 330, 360]  # prix fictifs en milliers d'euros
}

df = pd.DataFrame(data)

# Séparation des variables explicatives et de la variable cible
X = df[["Surface", "Chambres"]]
y = df["Prix"]

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# --- Interface Streamlit ---
st.title("Prédiction du Prix d'une Maison")

st.markdown("""
Ce programme utilise un modèle de régression linéaire entraîné sur un jeu de données fictif.
Entrez les caractéristiques de la maison ci-dessous pour obtenir une prédiction du prix.
""")

# Saisie interactive des caractéristiques
surface = st.number_input("Surface de la maison (en m²)", min_value=20, max_value=300, value=100)
chambres = st.number_input("Nombre de chambres", min_value=1, max_value=10, value=3)

# Prédiction si l'utilisateur clique sur le bouton
if st.button("Prédire le prix"):
    # Création du tableau de données pour la prédiction
    input_data = np.array([[surface, chambres]])
    prediction = model.predict(input_data)[0]
    st.success(f"Le prix prédit pour cette maison est de {prediction:.2f} milliers d'euros.")
    
# Affichage du jeu de données et du modèle pour information
st.subheader("Jeu de données d'entraînement (fictif)")
st.dataframe(df)
