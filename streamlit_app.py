import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Creer la page Streamlit
st.title('Prédiction de la valeur de CO2 pour un véhicule')

# Formulaire pour saisir les données du véhicule
modm = st.number_input("Masse (Kg)", min_value=700, max_value=5000, value=700)
carburant = st.selectbox("Carburant", options=['ES', 'GH', 'GO', 'EH', 'GP/ES', 'ES/GP', 'GN', 'GN/ES', 'ES/GN', 'FE'])
conso_mixte = st.number_input("Conso mixte (L/100km)", min_value=0.0, max_value=50.0, value=0.0)
puiss_admin = st.number_input("Puiss administrative (ch)", min_value=1, max_value=100, value=1)

# Champ de selection pour le modele
model_choice = st.selectbox("Choisissez un modèle", options=["Random Forest", "Gradient Boosting"])

# Charger les donnees
df = pd.read_csv("Data pour BI.csv")

# Selectionner les colonnes pertinentes
features = ['MODM', 'Carburant', 'Conso_mixte', 'Puiss_admin']
X = df[features]
y = df['CO2']

# Prétraiter les données
scaler = StandardScaler()
X[['MODM', 'Conso_mixte', 'Puiss_admin']] = scaler.fit_transform(X[['MODM', 'Conso_mixte', 'Puiss_admin']])

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X[['Carburant']])

X_encoded = encoder.transform(X[['Carburant']]).toarray()
X.drop('Carburant', axis=1, inplace=True)

X = np.concatenate((X, X_encoded), axis=1)

# Choisir le modèle
if model_choice == "Random Forest":
    model = RandomForestRegressor(random_state=42)
elif model_choice == "Gradient Boosting":
    model = GradientBoostingRegressor(random_state=42)

# Entraîner le modèle
model.fit(X, y)

# Enregistrer le modèle et les transformateurs préalablement entraînés
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')


# Fonction pour prédire la valeur de CO2

def predict_CO2(modm, carburant, conso_mixte, puiss_admin):
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    input_data = pd.DataFrame({
        'MODM': [modm],
        'Carburant': [carburant],
        'Conso_mixte': [conso_mixte],
        'Puiss_admin': [puiss_admin]
    })

    input_data[['MODM', 'Conso_mixte', 'Puiss_admin']] = scaler.transform(input_data[['MODM', 'Conso_mixte', 'Puiss_admin']])
    input_data_encoded = encoder.transform(input_data[['Carburant']]).toarray()
    input_data.drop('Carburant', axis=1, inplace=True)
    input_data = np.concatenate((input_data, input_data_encoded), axis=1)

    prediction = model.predict(input_data)
    return prediction[0]

# Afficher la prédiction
st.subheader('Résultat de la prédiction')
if st.button("Prédire"):
    prediction = predict_CO2(modm, carburant, conso_mixte, puiss_admin)
    st.write(f"La valeur prédite pour le CO2 est: {prediction:.2f} g/km")
