import os
import math
import joblib
import pandas as pd
from datetime import datetime

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

        self.DISPATCH_POLICY = {
            "officer_base": 2,
            "officer_multiplier": 15,
            "officer_max": 20,          
            "barricade_multiplier": 30,
            "barricade_max": 24         
        }

        self.HISTORICAL_AVERAGES = {
            "water_logging": 240,
            "public_event": 285,
            "procession": 210,
            "protest": 300,
            "construction": 360,
            "vehicle_breakdown": 120
        }

    def calculate_time_waves(self, hour, day_of_week):
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        day_sin = math.sin(2 * math.pi * day_of_week / 7.0)
        day_cos = math.cos(2 * math.pi * day_of_week / 7.0)
        return hour_sin, hour_cos, day_sin, day_cos

    def predict_event_impact(self, event_cause, zone, start_datetime_str, override_severity=None):
        dt = datetime.fromisoformat(start_datetime_str)
        hour = dt.hour
        day_of_week = dt.weekday()

        if override_severity is not None:
            severity_prob = override_severity
        elif not self.models_loaded:
            severity_prob = 0.5
        else:
            hour_sin, hour_cos, day_sin, day_cos = self.calculate_time_waves(hour, day_of_week)
            z_enc = self.le_zone.transform([zone])[0] if zone in self.le_zone.classes_ else 0
            c_enc = self.le_cause.transform([event_cause])[0] if event_cause in self.le_cause.classes_ else 0
            
            features = pd.DataFrame([[z_enc, c_enc, hour_sin, hour_cos, day_sin, day_cos]], 
                                    columns=['zone_encoded', 'cause_encoded', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'])
            severity_prob = float(self.model.predict(features)[0])

        peak_penalty = 1.4 if (8 <= hour <= 11) or (17 <= hour <= 21) else 1.0
        congestion_multiplier = max(0.1, 0.90 - (severity_prob * peak_penalty * 0.4))

        proactive_duration = int(severity_prob * 150) + 30 
        historical_baseline = self.HISTORICAL_AVERAGES.get(event_cause, 180)

        duration_without_astram = max(historical_baseline, proactive_duration + 45)

        officers_needed = min(
            int(self.DISPATCH_POLICY["officer_base"] + (severity_prob * self.DISPATCH_POLICY["officer_multiplier"])), 
            self.DISPATCH_POLICY["officer_max"]
        )
        
        barricades_count = min(
            int((severity_prob**2) * self.DISPATCH_POLICY["barricade_multiplier"]), 
            self.DISPATCH_POLICY["barricade_max"]
        )

        diversion_radius = round(severity_prob * 3.5, 1)

        return {
            "predictions": {
                "severity_index": round(severity_prob, 2),
                "predicted_speed_multiplier": round(congestion_multiplier, 2),
                "estimated_duration_minutes": proactive_duration,
                "duration_without_astram": duration_without_astram
            },
            "resource_allocation": {
                "recommended_officers": officers_needed,
                "barricades_count": barricades_count,
                "diversion_radius_km": diversion_radius
            }
        }