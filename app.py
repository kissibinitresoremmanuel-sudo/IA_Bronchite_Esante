import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
API_URL = "http://127.0.0.1:8000/predict"
from capteurs import lire_capteurs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="E-Santé Bronchite", layout="wide")
st.title("🏥 Système d'IA – Diagnostic Bronchite")

# Initialiser la session state pour les capteurs
if 'capteurs_data' not in st.session_state:
    st.session_state.capteurs_data = None

st.subheader("📡 Données capteurs (simulation ESP32)")

col_cap1, col_cap2 = st.columns([1, 1])
with col_cap1:
    if st.button("📥 Lire les capteurs", use_container_width=True):
        st.session_state.capteurs_data = lire_capteurs()
        st.success("✅ Capteurs lus avec succès !")

with col_cap2:
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.capteurs_data = None
        st.info("Données capteurs réinitialisées")

# Afficher les données capteurs si elles existent
if st.session_state.capteurs_data:
    capteurs = st.session_state.capteurs_data
    st.success("✅ Données capteurs chargées :")
    col_disp1, col_disp2, col_disp3, col_disp4 = st.columns(4)
    with col_disp1:
        st.metric("❤️ FC (bpm)", capteurs["frequence_cardiaque"])
    with col_disp2:
        st.metric("🫁 SpO2 (%)", capteurs["spo2"])
    with col_disp3:
        st.metric("🌡️ Tamb (°C)", capteurs["temperature_ambiante"])
    with col_disp4:
        st.metric("💧 Humidité (%)", capteurs["humidite"])
else:
    st.warning("⚠️ Aucune donnée capteur - Utilisez les valeurs par défaut ou lire les capteurs")

# Charger et entraîner le modèle
@st.cache_resource
def load_model():
    try:
        base = os.getcwd()
        dataset = os.path.join(base, "bronchite_cote_ivoire_dataset_1000.xlsx")
        
        if not os.path.exists(dataset):
            st.error(f"Fichier non trouvé : {dataset}")
            return None, None, None
        
        data = pd.read_excel(dataset)
        X = data.drop("bronchite", axis=1)
        y = data["bronchite"]
        
        # Remplir valeurs manquantes
        X = X.fillna(X.median(numeric_only=True))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Entraîner modèle
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=1))
        ])
        
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        
        return model, X.columns.tolist(), accuracy
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None, None

model, feature_names, accuracy = load_model()

if model is None:
    st.stop()

st.sidebar.metric("Précision du modèle", f"{accuracy:.2%}")

st.markdown("---")
st.subheader("📋 Remplissez les paramètres du patient")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Âge (ans)", 1, 100, 45)
    sexe = st.selectbox("Sexe", ["Femme (0)", "Homme (1)"], index=1)
    fumeur = st.selectbox("Fumeur", ["Non (0)", "Oui (1)"], index=0)
    annees_tabagisme = st.number_input("Années de tabagisme", 0, 80, 0)
    temperature_corporelle = st.number_input("Température corporelle (°C)", 35.0, 42.0, 37.0)

with col2:
    toux = st.slider("Intensité toux (0-3)", 0, 3, 0)
    essoufflement = st.slider("Essoufflement (0-3)", 0, 3, 0)
    fatigue = st.slider("Fatigue (0-3)", 0, 3, 0)
    douleur_thoracique = st.slider("Douleur thoracique (0-3)", 0, 3, 0)
    # Utiliser données capteur si disponibles, sinon valeur par défaut
    fc_default = st.session_state.capteurs_data["frequence_cardiaque"] if st.session_state.capteurs_data else 80
    frequence_cardiaque = st.number_input("Fréquence cardiaque (bpm)", 40, 200, fc_default)

col3, col4 = st.columns(2)

with col3:
    # Utiliser données capteur si disponibles, sinon valeur par défaut
    spo2_default = st.session_state.capteurs_data["spo2"] if st.session_state.capteurs_data else 95
    spo2 = st.number_input("SpO2 (%)", 70, 100, spo2_default)
    
    temp_amb_default = st.session_state.capteurs_data["temperature_ambiante"] if st.session_state.capteurs_data else 25
    temperature_ambiante = st.number_input("Température ambiante (°C)", 10, 50, temp_amb_default)

with col4:
    # Utiliser données capteur si disponibles, sinon valeur par défaut
    humid_default = st.session_state.capteurs_data["humidite"] if st.session_state.capteurs_data else 60
    humidite = st.number_input("Humidité (%)", 20, 100, humid_default)

st.markdown("---")

if st.button("Analyser"):

    sexe_val = 1 if sexe == "Homme" else 0
    fumeur_val = 1 if fumeur == "Oui" else 0

    payload = {
        "age": age,
        "sexe": sexe_val,
        "fumeur": fumeur_val,
        "annees_tabagisme": annees_tabagisme,
        "temperature_corporelle": temperature_corporelle,
        "toux": toux,
        "essoufflement": essoufflement,
        "fatigue": fatigue,
        "douleur_thoracique": douleur_thoracique,
        "frequence_cardiaque": frequence_cardiaque,
        "spo2": spo2,
        "temperature_ambiante": temperature_ambiante,
        "humidite": humidite
    }

    try:
        response = requests.post(API_URL, json=payload).json()

        if response.get("prediction_effectuee") is False:
            st.error("⚠️ Prédiction impossible")
            st.write("Capteurs défaillants :")
            for capteur in response["capteurs_defaillants"]:
                st.write("❌", capteur)
            st.info(response["action"])

        else:
            st.success(f"Probabilité de bronchite : {response['probabilite_bronchite']} %")
            st.write("Niveau de risque :", response["niveau_risque"])

    except Exception as e:
        st.error(f"Erreur lors de l'appel API : {e}")

st.markdown("---")
st.info("💡 **Disclaimer**: Ce système est un outil d'aide au diagnostic.")
