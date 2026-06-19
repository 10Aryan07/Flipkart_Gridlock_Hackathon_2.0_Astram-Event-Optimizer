from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from backend.inference import TrafficInferenceEngine

app = FastAPI(title="Gridlock ASTraM Optimizer API")
engine = TrafficInferenceEngine()

class EventRequest(BaseModel):
    event_name: str
    event_cause: str
    latitude: float
    longitude: float
    zone: str
    start_datetime: str
    override_severity: Optional[float] = None

@app.post("/simulate_event")
def simulate_event(req: EventRequest):
    try:
        dt = datetime.fromisoformat(req.start_datetime)
        impact = engine.predict_event_impact(req.event_cause, req.zone, dt.hour, dt.weekday())

        if req.override_severity is not None:
            impact["high_severity_prob"] = req.override_severity
            recalculated_multiplier = max(0.1, 0.90 - (req.override_severity * 1.4 * 0.4))
            impact["congestion_multiplier"] = round(recalculated_multiplier, 2)
            impact["estimated_duration_mins"] = int(req.override_severity * 350) + 45

        severity = impact["high_severity_prob"]

        officers_needed = int(2 + (severity * 15))
        barricades_count = int((severity ** 2) * 50) 
        requires_barricades = barricades_count > 5
        diversion_radius = round(severity * 3.5, 1)

        duration_with_astram = impact["estimated_duration_mins"]
        duration_without_astram = int(duration_with_astram * 1.55) 

        return {
            "status": "success",
            "event": req.event_name,
            "predictions": {
                "predicted_speed_multiplier": impact["congestion_multiplier"],
                "estimated_duration_minutes": duration_with_astram,
                "duration_without_astram": duration_without_astram,
                "severity_index": severity
            },
            "resource_allocation": {
                "recommended_officers": officers_needed,
                "barricades_count": barricades_count,
                "diversion_radius_km": diversion_radius
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "Active", "engine_ready": engine.models_loaded}