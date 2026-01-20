import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import json
from capteurs import lire_capteurs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# --- CONFIGURATION & SECURITE ---
API_URL = "https://ia-bronchite-esante.onrender.com/predict"
API_KEY = os.environ.get("API_KEY", "VOTRE_CLE_PAR_DEFAUT") 

st.set_page_config(page_title="DIGISANTE-APP3", layout="wide")
st.title("🏥 Système d'IA – Diagnostic Bronchite")

# --- GESTION DES ETATS ---
if 'capteurs_data' not in st.session_state:
    st.session_state.capteurs_data = None

# --- SECTION CAPTEURS ---
st.subheader("📡 Interface des Capteurs (ESP32)")

col_cap1, col_cap2 = st.columns([1, 1])
with col_cap1:
    if st.button("📥 Lire les données des capteurs", use_container_width=True):
        data = lire_capteurs()
        if data:
            st.session_state.capteurs_data = data
            st.success("✅ Données reçues du matériel !")
        else:
            st.error("❌ Échec de lecture : Vérifiez la connexion.")

with col_cap2:
    if st.button("🔄 Réinitialiser (Mode Manuel)", use_container_width=True):
        st.session_state.capteurs_data = None
        st.info("Passage en saisie manuelle.")

capteurs = st.session_state.capteurs_data
if capteurs:
    st.info("🚀 **Mode Automatique :** Champs capteurs verrouillés.")
else:
    st.warning("✍️ **Mode Manuel :** Saisie manuelle activée.")

# --- CHARGEMENT DU MODELE ---
@st.cache_resource
def load_model():
    try:
        dataset_path = "bronchite_cote_ivoire_dataset_1000.xlsx"
        if not os.path.exists(dataset_path):
            return None, None, None
        data = pd.read_excel(dataset_path)
        X = data.drop("bronchite", axis=1)
        y = data["bronchite"]
        X = X.fillna(X.median(numeric_only=True))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
        ])
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        return model, X.columns.tolist(), acc
    except Exception as e:
        st.error(f"Erreur modèle : {e}")
        return None, None, None

model, feature_names, accuracy = load_model()

# --- FORMULAIRE ---
st.markdown("---")
st.subheader("📋 Informations Patient")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Âge (ans)", 1, 120, 45)
    sexe = st.selectbox("Sexe", ["Femme", "Homme"], index=1)
    fumeur = st.selectbox("Fumeur", ["Non", "Oui"], index=0)
    annees_tabagisme = st.number_input("Années de tabagisme", 0, 80, 0)
    temp_corporelle = st.number_input("Température corporelle (°C)", 30.0, 45.0, 37.0)

with col2:
    toux = st.slider("Intensité toux (0-3)", 0, 3, 0)
    essoufflement = st.slider("Essoufflement (0-3)", 0, 3, 0)
    fatigue = st.slider("Fatigue (0-3)", 0, 3, 0)
    douleur_thoracique = st.slider("Douleur thoracique (0-3)", 0, 3, 0)

# --- LOGIQUE CAPTEURS ---
st.markdown("#### 🩺 Constantes Vitales")
c3, c4 = st.columns(2)

def get_sensor_value(key, default):
    if capteurs and key in capteurs and capteurs[key] is not None:
        return capteurs[key], True
    return default, False

val_fc, lock_fc = get_sensor_value("frequence_cardiaque", 80)
val_spo2, lock_spo2 = get_sensor_value("spo2", 98)
val_temp_amb, lock_temp_amb = get_sensor_value("temperature_ambiante", 25.0)
val_humid, lock_humid = get_sensor_value("humidite", 50)

with c3:
    frequence_cardiaque = st.number_input("Fréquence cardiaque (bpm)", 30, 220, int(val_fc), disabled=lock_fc)
    spo2 = st.number_input("Saturation SpO2 (%)", 50, 100, int(val_spo2), disabled=lock_spo2)

with c4:
    temperature_ambiante = st.number_input("Température Ambiante (°C)", 0.0, 60.0, float(val_temp_amb), disabled=lock_temp_amb)
    humidite = st.number_input("Humidité relative (%)", 0, 100, int(val_humid), disabled=lock_humid)

# --- ANALYSE ---
st.markdown("---")
if st.button("🚀 LANCER L'ANALYSE IA", use_container_width=True):
    sexe_val = 1 if sexe == "Homme" else 0
    fumeur_val = 1 if fumeur == "Oui" else 0

    payload = {
        "age": int(age), "sexe": sexe_val, "fumeur": fumeur_val,
        "annees_tabagisme": int(annees_tabagisme), "temperature_corporelle": float(temp_corporelle),
        "toux": int(toux), "essoufflement": int(essoufflement), "fatigue": int(fatigue),
        "douleur_thoracique": int(douleur_thoracique), "frequence_cardiaque": int(frequence_cardiaque),
        "spo2": int(spo2), "temperature_ambiante": float(temperature_ambiante), "humidite": int(humidite)
    }

    headers = {"Content-Type": "application/json", "x-api-key": API_KEY}

    try:
        with st.spinner("Analyse en cours..."):
            response = requests.post(API_URL, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                resultat = response.json()
                prob = float(resultat.get('probabilite_bronchite'))
                
                st.balloons()
                st.subheader("📊 Résultats du Diagnostic")
                
                # --- NOUVELLE LOGIQUE DE RISQUE (FAIBLE, MOYEN, ELEVE) ---
                if prob < 30:
                    st.success(f"### Risque : FAIBLE ({prob}%)")
                    st.write("✅ Les indicateurs sont rassurants.")
                elif 30 <= prob < 70:
                    st.warning(f"### Risque : MOYEN ({prob}%)")
                    st.write("⚠️ Une surveillance est nécessaire.")
                else:
                    st.error(f"### Risque : ÉLEVÉ ({prob}%)")
                    st.write("🚨 Risque important de bronchite détecté.")

                st.info(f"💡 **Conseil :** {resultat.get('action', 'Consultez un médecin pour confirmer ce résultat.')}")
                
            elif response.status_code == 403:
                st.error("❌ Erreur : Accès non autorisé (Clé API invalide).")
            else:
                st.error(f"❌ Erreur Serveur ({response.status_code})")
    except Exception as e:
        st.error(f"📡 Erreur de connexion : {e}")

st.sidebar.markdown(f"**Fiabilité du modèle :** {accuracy:.2%}")
