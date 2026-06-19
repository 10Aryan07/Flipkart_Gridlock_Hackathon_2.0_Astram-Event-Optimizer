import os
import math
import joblib
import pandas as pd

class TrafficInferenceEngine:
    def __init__(self):
        print("Booting ASTraM Intelligence Engine...")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)

        model_path = os.path.join(project_root, "models", "astram_rf_model.pkl")
        zone_enc_path = os.path.join(project_root, "models", "le_zone.pkl")
        cause_enc_path = os.path.join(project_root, "models", "le_cause.pkl")
        
        try:
            self.model = joblib.load(model_path)
            self.le_zone = joblib.load(zone_enc_path)
            self.le_cause = joblib.load(cause_enc_path)
            self.models_loaded = True
            print("ASTraM Random Forest Model Loaded Successfully.")
        except FileNotFoundError:
            print("Error: Model files not found in /models directory.")
            self.models_loaded = False

    def calculate_time_waves(self, hour, day_of_week):
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        day_sin = math.sin(2 * math.pi * day_of_week / 7.0)
        day_cos = math.cos(2 * math.pi * day_of_week / 7.0)
        return hour_sin, hour_cos, day_sin, day_cos

    def predict_event_impact(self, event_cause, zone, hour, day_of_week):
        if not self.models_loaded:
            return {"congestion_multiplier": 0.5, "estimated_duration_mins": 60, "high_severity_prob": 0.5}

        hour_sin, hour_cos, day_sin, day_cos = self.calculate_time_waves(hour, day_of_week)
    
        z_enc = self.le_zone.transform([zone])[0] if zone in self.le_zone.classes_ else 0
        c_enc = self.le_cause.transform([event_cause])[0] if event_cause in self.le_cause.classes_ else 0

        features = pd.DataFrame([[z_enc, c_enc, hour_sin, hour_cos, day_sin, day_cos]], 
                                columns=['zone_encoded', 'cause_encoded', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'])

        severity_prob = float(self.model.predict(features)[0])

        duration = int(severity_prob * 300) + 30
        base_multiplier = 0.90
        peak_penalty = 1.4 if (8 <= hour <= 11) or (17 <= hour <= 21) else 1.0
        congestion_multiplier = max(0.1, base_multiplier - (severity_prob * peak_penalty * 0.4))

        return {
            "congestion_multiplier": round(congestion_multiplier, 2),
            "estimated_duration_mins": duration,
            "high_severity_prob": round(severity_prob, 2)
        }