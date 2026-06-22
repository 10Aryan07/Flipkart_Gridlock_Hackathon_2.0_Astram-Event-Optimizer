from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.inference import TrafficInferenceEngine

app = FastAPI(title="ASTraM API", version="2.1")
engine = TrafficInferenceEngine()

class EventPayload(BaseModel):
    event_name: str
    event_cause: str
    latitude: float
    longitude: float
    zone: str
    start_datetime: str
    override_severity: Optional[float] = None

@app.get("/health")
def health_check():
    """Diagnostic endpoint to verify API and Model status."""
    return {
        "status": "online",
        "model_loaded": getattr(engine, 'models_loaded', False),
        "version": "2.1"
    }

@app.post("/simulate_event")
def simulate_event(payload: EventPayload):
    """Main inference endpoint with robust error handling."""
    try:
        result = engine.predict_event_impact(
            event_cause=payload.event_cause,
            zone=payload.zone,
            start_datetime_str=payload.start_datetime,
            override_severity=payload.override_severity
        )
        return result
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Data formatting error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Intelligence Engine Error: {str(e)}")