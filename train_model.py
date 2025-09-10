import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def create_training_data():
    """
    Crée des données de consommation électrique simulées pour l'entraînement.
    """
    np.random.seed(42)  # Pour des résultats reproductibles
    
    # Simuler 1000 points de données normales
    # La plupart des appareils consomment entre 40W et 100W
    normal_data = np.random.normal(loc=70, scale=25, size=1000)
    
    # S'assurer que les valeurs sont positives et formater pour le modèle
    normal_data = np.abs(normal_data).reshape(-1, 1)
    
    print(f"Données d'entraînement créées : {len(normal_data)} échantillons.")
    return normal_data

def train_anomaly_model():
    """
    Entraîne le modèle de détection d'anomalies et le sauvegarde.
    """
    print("=== Début de l'entraînement du modèle d'anomalie ===")
    
    # Obtenir les données d'entraînement
    training_data = create_training_data()
    
    # Créer le modèle "IsolationForest"
    # contamination=0.05 signifie qu'on s'attend à environ 5% d'anomalies
    model = IsolationForest(contamination=0.05, random_state=42)
    
    print("Entraînement du modèle en cours...")
    model.fit(training_data)
    
    # Créer le dossier 'models' s'il n'existe pas
    os.makedirs('models', exist_ok=True)
    
    # Définir le chemin pour sauvegarder le modèle
    model_path = 'models/anomaly_model.joblib'
    
    # Sauvegarder le modèle sur le disque
    joblib.dump(model, model_path)
    
    print(f"✅ Modèle sauvegardé avec succès dans : {model_path}")
    
    # Test rapide pour vérifier que le modèle fonctionne comme prévu
    test_normal = [[75]]   # Une valeur normale
    test_anomaly = [[500]] # Une valeur anormalement élevée
    
    prediction_normal = model.predict(test_normal)
    prediction_anomaly = model.predict(test_anomaly)
    
    print(f"\n--- Test rapide du modèle ---")
    print(f"Prédiction pour une valeur normale (75W) : {'Normal' if prediction_normal[0] == 1 else 'Anomalie'}")
    print(f"Prédiction pour une valeur élevée (500W) : {'Anomalie Détectée' if prediction_anomaly[0] == -1 else 'Non Détectée'}")
    print("---------------------------")


if __name__ == "__main__":
    # Cette condition assure que le code ne s'exécute que si le script est lancé directement
    train_anomaly_model()

