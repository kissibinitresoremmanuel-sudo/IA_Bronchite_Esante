import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def main():
    base = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    dataset = os.path.join(base, "bronchite_cote_ivoire_dataset_1000.xlsx")

    if not os.path.exists(dataset):
        print(f"Fichier introuvable : {dataset}")
        print("Placez le fichier Excel dans le même dossier que ce script.")
        return

    data = pd.read_excel(dataset)
    print("Données chargées :", data.shape)

    if "bronchite" not in data.columns:
        print("La colonne 'bronchite' est absente. Colonnes disponibles :", list(data.columns))
        return

    X = data.drop("bronchite", axis=1)
    y = data["bronchite"]

    # Remplir les valeurs manquantes
    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Pipeline simple avec normalisation et modèle
    models = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=1))
        ]),
    }
    
    if XGB_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(n_estimators=200, max_depth=3, random_state=42, tree_method='hist'))
        ])

    print("\n=== Résultats d'entraînement ===")
    best_model = None
    best_score = 0
    best_name = None

    for name, model in models.items():
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            # Entraîner le modèle complet
            model.fit(X_train, y_train)
            
            # Évaluation test
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            print(f"{name} - Test Accuracy: {acc:.4f}")

            if acc > best_score:
                best_score = acc
                best_model = model
                best_name = name
        except Exception as e:
            print(f"{name} - Erreur : {e}")

    print(f"\nMeilleur modèle : {best_name} (Accuracy: {best_score:.4f})")
    
    # Rapport complet
    if best_model is not None:
        pred = best_model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, pred))

        # ROC AUC si possible
        try:
            proba = best_model.predict_proba(X_test)
            auc = roc_auc_score(y_test, proba[:, 1])
            print(f"ROC AUC: {auc:.4f}")
        except Exception:
            pass

        # Test exemple première ligne
        try:
            proba = best_model.predict_proba(X_test.iloc[[0]])
            print(f"Probabilité de bronchite (première ligne test): {round(proba[0][1]*100, 2)} %")
        except Exception:
            pass

        # Test volontairement malade (cas artificiel)
        try:
            cas_malade = [[
                55,  # age
                1,   # sexe (homme)
                1,   # fumeur
                25,  # annees tabagisme
                39.2,# temperature corporelle
                3,   # toux sévère
                3,   # essoufflement
                3,   # fatigue
                2,   # douleur thoracique
                115, # fréquence cardiaque
                88,  # spo2 bas
                33,  # température ambiante
                85   # humidité
            ]]

            # construire DataFrame avec mêmes colonnes que X
            cols = X.columns.tolist()
            df_malade = pd.DataFrame(cas_malade, columns=cols)

            proba_malade = best_model.predict_proba(df_malade)
            print(f"\nCas malade – Probabilité de bronchite : {round(proba_malade[0][1]*100, 2)} %")
        except Exception as e:
            print(f"Erreur test cas_malade : {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Erreur durant l'exécution :", e)
        raise
