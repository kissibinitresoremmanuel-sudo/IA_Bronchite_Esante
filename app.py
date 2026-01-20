import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import json

# Remplace par ton URL Render en production
API_URL = "https://ia-bronchite-esante.onrender.com/predict"

from capteurs import lire_capteurs

# ==================================
# FONCTION DE VERROUILLAGE DYNAMIQUE
# ==================================
def champ_capteur_intelligent(label, cle_capteur, min_val, max_val, default_val):
    """
    Verrouille le champ si la donnée existe dans le dictionnaire des capteurs.
    """
    donnees_capteurs = st.session_state.get("capteurs_data")
    
    if donnees_capteurs and cle_capteur in donnees_capteurs and donnees_capteurs[cle_capteur] is not None:
        valeur_auto = float(donnees_capteurs[cle_capteur])
        st.info(f"✅ {label} : Reçu automatiquement")
        return st.number_input(label, value=valeur_auto, disabled=True)
    else:
        st.warning(f"✍️ {label} : Saisie manuelle requise")
        return st.number_input(label, min_value=float(min_val), max_value=float(max_val), value=float(default_val))

# =======================
# CONFIG STREAMLIT
# =======================
st.set_page_config(page_title="E-Santé Bronchite", layout="wide")
st.title("🏥 Système d'IA – Diagnostic de la Bronchite")

st.sidebar.header("🔐 Authentification")
# On essaie de récupérer la clé depuis l'environnement, sinon on demande la saisie
env_api_key = os.environ.get("API_KEY", "")
api_key = st.sidebar.text_input("Clé API", value=env_api_key, type="password")

# =======================
# SESSION CAPTEURS
# =======================
if "capteurs_data" not in st.session_state:
    st.session_state.capteurs_data = None

st.subheader("📡 Interface Matérielle (ESP32)")

colA, colB = st.columns(2)
with colA:
    if st.button("📥 LIRE LES CAPTEURS", use_container_width=True):
        with st.spinner("Lecture du matériel..."):
            st.session_state.capteurs_data = lire_capteurs()
            if st.session_state.capteurs_data:
                st.success("Données synchronisées !")
            else:
                st.error("Erreur de connexion matériel.")

with colB:
    if st.button("🔄 RÉINITIALISER / SAISIE MANUELLE", use_container_width=True):
        st.session_state.capteurs_data = None
        st.rerun()

# =======================
# FORMULAIRE PATIENT
# =======================
st.markdown("---")
st.subheader("👤 Informations Patient & Cliniques")

colL, colR = st.columns(2)
with colL:
    age = st.number_input("Âge", 1, 100, 45)
    sexe = st.selectbox("Sexe", ["Femme", "Homme"])
    fumeur = st.selectbox("Fumeur", ["Non", "Oui"])
    annees_tabagisme = st.number_input("Années de tabagisme", 0, 80, 0)
    temp_corporelle = st.number_input("Température corporelle (°C)", 35.0, 42.0, 37.0)

with colR:
    toux = st.slider("Toux (0-3)", 0, 3, 0)
    essoufflement = st.slider("Essoufflement (0-3)", 0, 3, 0)
    fatigue = st.slider("Fatigue (0-3)", 0, 3, 0)
    douleur_thoracique = st.slider("Douleur thoracique (0-3)", 0, 3, 0)

# =======================================
# SECTION PHYSIOLOGIQUE (AUTO vs MANUEL)
# =======================================
st.markdown("---")
st.subheader("🔌 Constantes Physiologiques")
col1, col2 = st.columns(2)

with col1:
    frequence_cardiaque = champ_capteur_intelligent("Fréquence cardiaque (bpm)", "frequence_cardiaque", 40, 200, 80)
    spo2 = champ_capteur_intelligent("Saturation SpO2 (%)", "spo2", 70, 100, 98)

with col2:
    temperature_ambiante = champ_capteur_intelligent("Température ambiante (°C)", "temperature_ambiante", 10, 50, 25)
    humidite = champ_capteur_intelligent("Humidité (%)", "humidite", 20, 100, 50)

# =======================
# ANALYSE IA
# =======================
st.markdown("---")
if st.button("🧠 LANCER L'ANALYSE DIAGNOSTIQUE", use_container_width=True, type="primary"):
    if not api_key:
        st.error("Veuillez saisir la clé API dans la barre latérale.")
    else:
        # On utilise bien temp_corporelle définie plus haut
        payload = {
            "age": age,
            "sexe": 1 if sexe == "Homme" else 0,
            "fumeur": 1 if fumeur == "Oui" else 0,
            "annees_tabagisme": annees_tabagisme,
            "temperature_corporelle": float(temp_corporelle),
            "toux": toux,
            "essoufflement": essoufflement,
            "fatigue": fatigue,
            "douleur_thoracique": douleur_thoracique,
            "frequence_cardiaque": frequence_cardiaque,
            "spo2": spo2,
            "temperature_ambiante": temperature_ambiante,
            "humidite": humidite
        }

        headers = {"x-api-key": api_key}

        try:
            with st.spinner("L'IA analyse vos données..."):
                response = requests.post(API_URL, json=payload, headers=headers)
                
                if response.status_code == 200:
                    res = response.json()
                    prob = float(res.get('probabilite_bronchite', 0))
                    
                    st.balloons()
                    st.subheader("📊 Résultats du Diagnostic")
                    
                    # Logique d'affichage par paliers de risque
                    if prob < 30:
                        st.success(f"### PRÉDICTION : FAIBLE ({prob}%)")
                        st.write("✅ Risque de bronchite très limité selon les paramètres actuels.")
                    elif 30 <= prob < 60:
                        st.warning(f"### PRÉDICTION : MOYEN ({prob}%)")
                        st.write("⚠️ Risque modéré. Une surveillance clinique est recommandée.")
                    else:
                        st.error(f"### PRÉDICTION : ÉLEVÉ ({prob}%)")
                        st.write("🚨 Risque important détecté. Une consultation médicale est urgente.")

                    st.info(f"💡 **Action recommandée :** {res.get('action', 'Consultez un médecin.')}")
                
                elif response.status_code == 403:
                    st.error("❌ Accès refusé : Clé API incorrecte.")
                else:
                    st.error(f"Erreur {response.status_code} : Serveur indisponible.")
        except Exception as e:
            st.error(f"Erreur de connexion : {e}")
