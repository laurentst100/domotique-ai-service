import os
import joblib
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# 1. Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)

# 2. Charger le modèle d'IA au démarrage
try:
    anomaly_model = joblib.load('models/anomaly_model.joblib')
    print("✅ Modèle 'anomaly_model.joblib' chargé avec succès.")
except Exception as e:
    anomaly_model = None
    print(f"❌ Erreur lors du chargement du modèle : {e}")

# 3. Endpoint de santé
@app.route('/health', methods=['GET'])
def health_check():
    if anomaly_model:
        return jsonify({"status": "operational", "model_status": "loaded"})
    else:
        return jsonify({"status": "degraded", "model_status": "not_loaded"}), 500

# 4. Endpoint d'analyse d'anomalie
@app.route('/api/analyze/anomaly', methods=['POST'])
def analyze_anomaly():
    if not anomaly_model:
        return jsonify({"error": "Le modèle d'analyse n'est pas disponible."}), 503
    data = request.get_json()
    if not data or 'power' not in data or 'deviceId' not in data:
        return jsonify({"error": "Les champs 'power' et 'deviceId' sont requis."}), 400
    try:
        power_value = float(data['power'])
        device_id = data['deviceId']
        power_array = np.array([[power_value]])
        prediction = anomaly_model.predict(power_array)
        is_anomaly = bool(prediction[0] == -1)  # <-- Correction ici
        anomaly_score = float(anomaly_model.decision_function(power_array)[0])
        response = {
            "deviceId": device_id,
            "power_consumption": power_value,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "analysis_timestamp": datetime.now().isoformat()
        }
        return jsonify(response)
    except ValueError:
        return jsonify({"error": "La valeur de 'power' doit être un nombre."}), 400
    except Exception as e:
        return jsonify({"error": f"Erreur interne du serveur : {e}"}), 500

# 5. Démarrer le serveur sur le port 8000 UNIQUEMENT
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
