from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Header, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from capteurs import lire_capteurs

# Clé API secrète (lue depuis la variable d'environnement)
API_KEY_SECRET = os.environ.get("API_KEY")
if not API_KEY_SECRET:
    raise RuntimeError("API_KEY non définie : définissez la variable d'environnement API_KEY pour l'API.")

# Historique global des capteurs
historique_capteurs = {
    "frequence_cardiaque": [],
    "spo2": [],
    "temperature_ambiante": [],
    "humidite": []
}
# 1. Initialiser l'API
app = FastAPI(
    title="API IA – Prédiction Bronchite",
    version="1.0.0",
    description="API pour le diagnostic de bronchite avec données capteurs"
)

# Ajouter CORS pour permettre les requêtes depuis Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Charger et entraîner le modèle
@app.on_event("startup")
def load_model():
    global model, feature_names, accuracy
    try:
        base = os.getcwd()
        dataset = os.path.join(base, "bronchite_cote_ivoire_dataset_1000.xlsx")
        
        data = pd.read_excel(dataset)
        X = data.drop("bronchite", axis=1)
        y = data["bronchite"]
        
        X = X.fillna(X.median(numeric_only=True))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=1))
        ])
        
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        feature_names = X.columns.tolist()
        
        print(f"✅ Modèle chargé - Précision: {accuracy:.2%}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")

# Fonction pour vérifier les capteurs défaillants
def verifier_capteurs(donnees):
    def verifier(nom, valeur):
        historique_capteurs[nom].append(valeur is not None)
        if len(historique_capteurs[nom]) > 3:
            historique_capteurs[nom].pop(0)
        return all(v is False for v in historique_capteurs[nom])

    capteurs_defaillants = []

    if verifier("frequence_cardiaque", donnees.frequence_cardiaque):
        capteurs_defaillants.append("Capteur fréquence cardiaque (MAX30100)")

    if verifier("spo2", donnees.spo2):
        capteurs_defaillants.append("Capteur SpO2 (MAX30100)")

    if verifier("temperature_ambiante", donnees.temperature_ambiante):
        capteurs_defaillants.append("Capteur température ambiante (DHT22)")

    if verifier("humidite", donnees.humidite):
        capteurs_defaillants.append("Capteur humidité (DHT22)")

    return capteurs_defaillants

# Fonction pour vérifier les incohérences des données
def verifier_incoherences(donnees):
    incoherences = []

    if donnees.frequence_cardiaque is not None and not (40 <= donnees.frequence_cardiaque <= 180):
        incoherences.append("Fréquence cardiaque incohérente")

    if donnees.spo2 is not None and not (80 <= donnees.spo2 <= 100):
        incoherences.append("SpO2 incohérente")

    if not (35 <= donnees.temperature_corporelle <= 42):
        incoherences.append("Température corporelle incohérente")

    if not (0 <= donnees.toux <= 3):
        incoherences.append("Toux incohérente")

    if not (0 <= donnees.essoufflement <= 3):
        incoherences.append("Essoufflement incohérent")

    if not (0 <= donnees.fatigue <= 3):
        incoherences.append("Fatigue incohérente")

    if not (0 <= donnees.douleur_thoracique <= 3):
        incoherences.append("Douleur thoracique incohérente")

    if donnees.temperature_ambiante is not None and not (20 <= donnees.temperature_ambiante <= 40):
        incoherences.append("Température ambiante incohérente")

    if donnees.humidite is not None and not (30 <= donnees.humidite <= 100):
        incoherences.append("Humidité incohérente")

    return incoherences

# Fonction pour journaliser les capteurs défaillants
def journaliser(capteurs):
    with open("maintenance.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - Capteurs défaillants : {', '.join(capteurs)}\n")

# 3. Définir le format des données entrantes
class DonneesPatient(BaseModel):
    age: int
    sexe: int
    fumeur: int
    annees_tabagisme: int
    temperature_corporelle: float
    toux: int
    essoufflement: int
    fatigue: int
    douleur_thoracique: int
    frequence_cardiaque: int
    spo2: int
    temperature_ambiante: int
    humidite: int

class ResultatDiagnostic(BaseModel):
    prediction: int
    probabilite_sain: float
    probabilite_bronchite: float
    diagnostic: str
    recommandation: str

# 4. Routes API
@app.get("/")
def root():
    return {
        "message": "API Diagnostic Bronchite",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "Documentation interactive (Swagger UI)",
            "/redoc": "Documentation ReDoc",
            "/predict": "POST - Prédire la bronchite",
            "/capteurs": "GET - Lire les capteurs ESP32",
            "/health": "GET - Vérifier l'état du serveur"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "precision_model": f"{accuracy:.2%}"}

@app.get("/capteurs")
def get_capteurs():
    """Lire les données des capteurs ESP32"""
    try:
        capteurs = lire_capteurs()
        return {
            "status": "success",
            "data": capteurs
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/test-defaillance")
def test_defaillance():
    """Simuler des capteurs défaillants pour test (MAX30100 et DHT22)"""
    # Simuler 3 appels avec capteurs défaillants
    for _ in range(3):
        historique_capteurs["frequence_cardiaque"].append(False)
        historique_capteurs["spo2"].append(False)
        historique_capteurs["temperature_ambiante"].append(False)
        historique_capteurs["humidite"].append(False)
    
    return {
        "message": "Capteurs défaillants simulés - Prochains appels /predict les détecteront",
        "historique": historique_capteurs
    }

@app.post("/predict")
def predire_bronchite(donnees: DonneesPatient, x_api_key: str = Header(None)):
    """Prédire la bronchite basé sur les données patient et capteurs"""
    # Vérifier la clé API
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=403, detail="Accès interdit : Clé API invalide")

    try:
        # Vérifier les capteurs défaillants
        capteurs_defaillants = verifier_capteurs(donnees)
        
        if capteurs_defaillants:
            journaliser(capteurs_defaillants)
            return {
                "prediction_effectuee": False,
                "capteurs_defaillants": capteurs_defaillants,
                "action": "Vérifiez l'état des capteurs et réessayez"
            }
        
        # Vérifier les incohérences des valeurs
        incoherences = verifier_incoherences(donnees)
        if incoherences:
            return {
                "erreur": "Valeurs incohérentes détectées",
                "incoherences": incoherences,
                "prediction_effectuee": False,
                "action": "Corriger les valeurs et vérifier les capteurs"
            }
        
        # Faire la prédiction
        input_data = pd.DataFrame([[
            donnees.age,
            donnees.sexe,
            donnees.fumeur,
            donnees.annees_tabagisme,
            donnees.temperature_corporelle,
            donnees.toux,
            donnees.essoufflement,
            donnees.fatigue,
            donnees.douleur_thoracique,
            donnees.frequence_cardiaque,
            donnees.spo2,
            donnees.temperature_ambiante,
            donnees.humidite
        ]], columns=feature_names)
        
        proba = model.predict_proba(input_data)
        prediction = model.predict(input_data)[0]
        
        prob_sain = float(proba[0][0] * 100)
        prob_bronchite = float(proba[0][1] * 100)
        
        # Calculer le niveau de risque
        niveau_risque = "faible" if prob_bronchite < 40 else ("modéré" if prob_bronchite < 70 else "élevé")
        
        return {
            "prediction_effectuee": True,
            "prediction": int(prediction),
            "probabilite_sain": prob_sain,
            "probabilite_bronchite": prob_bronchite,
            "niveau_risque": niveau_risque,
            "diagnostic": "Risque de bronchite détecté" if prediction == 1 else "Pas de bronchite détectée",
            "recommandation": "Consulter un professionnel médical" if prediction == 1 else "Continuer la surveillance"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)