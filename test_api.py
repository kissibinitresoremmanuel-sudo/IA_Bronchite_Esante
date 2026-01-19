#!/usr/bin/env python3
import requests
import json

API_URL = "http://127.0.0.1:8000/predict"

# Test avec valeurs incohérentes
payload = {
    "age": 50,
    "sexe": 1,
    "fumeur": 1,
    "annees_tabagisme": 20,
    "temperature_corporelle": 50,  # Incohérente (max 42)
    "toux": 5,  # Incohérente (max 3)
    "essoufflement": 3,
    "fatigue": 2,
    "douleur_thoracique": 1,
    "frequence_cardiaque": 10,  # Incohérente (min 40)
    "spo2": 200,  # Incohérente (max 100)
    "temperature_ambiante": 15,  # Incohérente (min 20)
    "humidite": 110  # Incohérente (max 100)
}

print("🧪 Test API /predict avec valeurs incohérentes")
print("=" * 60)
print(f"Payload: {json.dumps(payload, indent=2)}")
print("=" * 60)

try:
    response = requests.post(API_URL, json=payload)
    result = response.json()
    print(f"✅ Réponse reçue:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"❌ Erreur: {e}")

# Test avec valeurs cohérentes
print("\n\n🧪 Test API /predict avec valeurs cohérentes")
print("=" * 60)

payload_ok = {
    "age": 50,
    "sexe": 1,
    "fumeur": 1,
    "annees_tabagisme": 20,
    "temperature_corporelle": 37.5,  # ✅ Cohérent
    "toux": 2,  # ✅ Cohérent
    "essoufflement": 1,
    "fatigue": 2,
    "douleur_thoracique": 1,
    "frequence_cardiaque": 75,  # ✅ Cohérent
    "spo2": 95,  # ✅ Cohérent
    "temperature_ambiante": 25,  # ✅ Cohérent
    "humidite": 60  # ✅ Cohérent
}

print(f"Payload: {json.dumps(payload_ok, indent=2)}")
print("=" * 60)

try:
    response = requests.post(API_URL, json=payload_ok)
    result = response.json()
    print(f"✅ Réponse reçue:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"❌ Erreur: {e}")
